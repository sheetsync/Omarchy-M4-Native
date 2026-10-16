#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Produce hash-only evidence for the J713 owner-filter baseline on macOS."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.request

FRACTIONAL_BITS = (16, 20, 24, 28, 32, 36, 40, 44, 48, 52)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def command(arguments, **options):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=300, **options)
    stage = Path(arguments[-1]).name
    require(result.returncode == 0,
            f'Diagnostic subprocess failed ({stage}, exit {result.returncode}): '
            + result.stderr[-4000:].strip())
    return result.stdout.strip()


def load_owner(root):
    specification = importlib.util.spec_from_file_location('fir_owner', root / 'tools/m4-j713-fir-owner.py')
    owner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(owner)
    return owner


def run_stage(source, dependencies, workspace, name):
    bootstrap = 'import runpy,sys; sys.path[:0]=sys.argv[1:3]; runpy.run_path(sys.argv[3],run_name="__main__")'
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('PYTHON', 'DYLD_'))}
    environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PATH='/usr/bin:/bin')
    command([sys.executable, '-I', '-B', '-c', bootstrap, str(source), str(dependencies),
             str(source / name)], cwd=workspace, env=environment)


def wav_samples(data):
    require(data[:4] == b'RIFF' and data[8:12] == b'WAVE', 'Unexpected WAV container')
    offset = 12
    while offset + 8 <= len(data):
        chunk, size = struct.unpack_from('<4sI', data, offset)
        payload = data[offset + 8:offset + 8 + size]
        require(len(payload) == size, 'Truncated WAV chunk')
        if chunk == b'data':
            return payload
        offset += 8 + size + size % 2
    raise RuntimeError('Missing WAV samples')


def summarize_samples(path, numpy, frames=None):
    data = path.read_bytes()
    samples = numpy.frombuffer(wav_samples(data) if path.suffix == '.wav' else data, dtype='<f4')
    require(samples.size > 0 and numpy.isfinite(samples).all(), 'Invalid sample array')
    values = samples.astype(numpy.float64)
    quantized = {}
    for bits in FRACTIONAL_BITS:
        integers = numpy.rint(numpy.ldexp(values, bits))
        require(numpy.isfinite(integers).all() and
                numpy.all(integers >= -(2 ** 63)) and numpy.all(integers < 2 ** 63),
                'Quantized sample exceeds int64')
        quantized[str(bits)] = digest(integers.astype('<i8').tobytes())
    result = {'sha256': digest(data), 'size': len(data), 'samples': int(samples.size),
              'peak': float(numpy.max(numpy.abs(values))),
              'l1': float(numpy.sum(numpy.abs(values))), 'quantized_int64_sha256': quantized}
    if frames:
        require(samples.size % frames == 0, 'Invalid channel layout')
        channels = values.reshape(-1, frames)
        result['channels'] = [{'peak': float(numpy.max(numpy.abs(channel))),
                               'l1': float(numpy.sum(numpy.abs(channel)))} for channel in channels]
        result['frames_per_channel'] = frames
    else:
        result['taps'] = int(samples.size)
    return result


def convolve_coefficients(first, second):
    """Sum scalar coefficient products with compensation instead of an FFT."""
    left = tuple(float(value) for value in first)
    right = tuple(float(value) for value in second)
    return [math.fsum(left[index] * right[position - index]
                     for index in range(max(0, position - len(right) + 1),
                                        min(len(left), position + 1)))
            for position in range(len(left) + len(right) - 1)]


def summarize_scalar(workspace, source, numpy):
    sys.path.insert(0, str(source))
    from analyze_transfer import AMP, IMPULSE, load, select_length, wav_float32
    from qualify_memory import float_coeff
    memory = workspace / 'native-memory'
    rows = {row['name']: row for row in json.loads((memory / 'memory-runs.json').read_text())}
    highpasses = []
    for label in ('hp90', 'hp80'):
        response = numpy.asarray(load(rows[label + '-amp0.0001-lane0'], memory)[0][IMPULSE:]) / AMP
        length, _ = select_length(response)
        highpasses.append(float_coeff(response[:length]))
    combined = convolve_coefficients(*highpasses)
    transfer = workspace / 'native-transfer'
    transfers = json.loads((transfer / 'transfers.json').read_text())
    directory = workspace / 'scalar-output'
    directory.mkdir()
    result = {}
    for lane in range(4):
        row = next(row for row in transfers
                   if row['bypass'] == 0 and row['amplitude'] < 0.00015 and row['lane'] == lane)
        response = numpy.asarray(load(row, transfer)[lane][IMPULSE:]) / AMP
        length, _ = select_length(response)
        tone = float_coeff(response[:length])
        full = numpy.asarray(convolve_coefficients(combined, tone))
        length, _ = select_length(full)
        coefficients = float_coeff(full[:length])
        name = f'j713-static-minimal-lane{lane}-48000.wav'
        path = directory / name
        wav_float32(path, coefficients)
        baseline = numpy.frombuffer(wav_samples((workspace / 'output' / name).read_bytes()),
                                    dtype='<f4').astype(numpy.float64)
        require(len(baseline) == len(coefficients), 'Scalar tap count differs')
        difference = numpy.abs(baseline - coefficients)
        result[name] = {**summarize_samples(path, numpy),
                        'versus_fft': {'different_samples': int(numpy.count_nonzero(difference)),
                                       'maximum_absolute_difference': float(numpy.max(difference)),
                                       'sum_absolute_differences': float(numpy.sum(difference))}}
    return result


def diagnose(root, workspace):
    require(platform.system() == 'Darwin' and platform.machine() == 'arm64', 'Native arm64 macOS required')
    require(sys.version_info[:3] == (3, 13, 9) and not sys.flags.optimize, 'CPython 3.13.9 required')
    version = command(['/usr/bin/sw_vers', '-productVersion'])
    build = command(['/usr/bin/sw_vers', '-buildVersion'])
    require((version, build) == ('26.6.2', '25G83'), 'Runner is not the qualified 26.6.2/25G83 baseline')
    owner = load_owner(root)
    bundle = root / 'audio/j713/owner-fir'
    source = bundle / 'src'
    manifest = json.loads((bundle / 'manifest.json').read_bytes())
    source_hashes = {}
    for name, pin in manifest['sources'].items():
        source_hashes[name] = digest(owner.checked_bytes(source / name, pin))
    inputs = {}
    (workspace / 'source').mkdir()
    for name, record in manifest['owner_inputs'].items():
        data, matched = owner.qualified_input(record)
        inputs[name] = {'sha256': digest(data), 'size': len(data)}
        if name != 'AudioDSP':
            (workspace / 'source' / name).write_bytes(data)
    (workspace / 'owner-inputs.json').write_text(json.dumps({name: item['sha256'] for name, item in inputs.items()}))
    helpers = workspace / 'helpers'
    command(['/bin/bash', str(bundle / 'build-helpers.sh'), str(helpers)])
    helper_hashes = {}
    for name in owner.HELPER_NAMES:
        helper_hashes[name] = digest((helpers / name).read_bytes())
        shutil.copy2(helpers / name, workspace / name)
    wheel = workspace / manifest['numpy']['filename']
    with urllib.request.urlopen(manifest['numpy']['url'], timeout=60) as response:
        data = response.read(manifest['numpy']['size'] + 1)
    require(len(data) == manifest['numpy']['size'] and digest(data) == manifest['numpy']['sha256'],
            'NumPy wheel identity mismatch')
    wheel.write_bytes(data)
    dependencies = workspace / 'dependencies'
    owner.unpack_wheel(wheel, dependencies)
    sys.path.insert(0, str(dependencies))
    import numpy
    require(numpy.__version__ == '2.5.3', 'Unexpected NumPy version')
    run_stage(source, dependencies, workspace, 'inspect_strip.py')
    for name, pin in manifest['presets'].items():
        owner.checked_bytes(workspace / 'decoded' / name, pin)
    (workspace / 'output').mkdir()
    run_stage(source, dependencies, workspace, 'run-transfer.py')
    (workspace / 'output').rename(workspace / 'native-transfer')
    (workspace / 'output').mkdir()
    run_stage(source, dependencies, workspace, 'run-memory.py')
    (workspace / 'output').rename(workspace / 'native-memory')
    run_stage(source, dependencies, workspace, 'qualify_memory.py')
    arrays = {str(path.relative_to(workspace)): summarize_samples(path, numpy, 73728)
              for directory in ('native-transfer', 'native-memory')
              for path in sorted((workspace / directory).glob('*.f32'))}
    outputs = {}
    for name, pin in manifest['outputs'].items():
        summary = summarize_samples(workspace / 'output' / name, numpy)
        summary['native_pin_match'] = (summary['size'], summary['sha256']) == (pin['size'], pin['sha256'])
        outputs[name] = summary
    report = json.loads((workspace / 'output/qualified-memory.json').read_text())
    for name, record in manifest['owner_inputs'].items():
        data, matched = owner.qualified_input(record)
        require(inputs[name] == {'sha256': digest(data), 'size': len(data)}, 'Input changed during diagnostic')
    return {'schema': 1, 'purpose': 'baseline numerical diagnosis; no playback or installation',
            'macos': {'version': version, 'build': build}, 'python': platform.python_version(),
            'numpy': numpy.__version__, 'owner_inputs': inputs, 'source_sha256': source_hashes,
            'helper_sha256': helper_hashes, 'compiler': command(['/usr/bin/clang', '--version']),
            'quantization': 'round-to-nearest-even(sample * 2**bits), little-endian signed int64',
            'outputs': outputs, 'raw_arrays': arrays,
            'scalar_outputs': summarize_scalar(workspace, source, numpy),
            'max_native_model_relative_error': max(row['relative_peak_error'] for row in report['native_memory_comparisons'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='j713-baseline-') as temporary:
        summary = diagnose(root, Path(temporary))
    serialized = json.dumps(summary, indent=2, sort_keys=True) + '\n'
    arguments.summary.parent.mkdir(parents=True, exist_ok=True)
    arguments.summary.write_text(serialized)
    print(serialized, end='')


if __name__ == '__main__':
    main()
