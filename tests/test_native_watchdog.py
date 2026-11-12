"""Compile the actual native watchdog without DDS or a motor interface."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_native_command_timeout_and_late_recovery(tmp_path):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.fail('A C++17 compiler is required for the native watchdog test')
    source = tmp_path / 'watchdog.cpp'
    source.write_text('''
#include "command_watchdog.hpp"
#include <cassert>
int main() {
    using namespace std::chrono_literals;
    auto t = CommandWatchdog::Clock::time_point{};
    CommandWatchdog w;
    assert(!w.CanPublish(t));
    assert(!w.fault());
    assert(w.Accept(t));
    for (int i = 1; i <= 50; ++i) {
        assert(w.CanPublish(t + i * 20ms));
        assert(w.Accept(t + i * 20ms));
    }
    assert(w.CanPublish(t + 1099ms));
    assert(!w.CanPublish(t + 1100ms));
    assert(w.fault());
    assert(!w.Accept(t + 1101ms));
    assert(!w.CanPublish(t + 1102ms));
    CommandWatchdog late;
    assert(late.Accept(t));
    assert(!late.Accept(t + 100ms)); // Producer races the writer after the deadline.
    assert(!late.CanPublish(t + 101ms));
    CommandWatchdog restarted;
    assert(restarted.Accept(t + 1000ms));
    assert(restarted.CanPublish(t + 1001ms));
}
''')
    exe = tmp_path / 'watchdog'
    subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-I', str(ROOT / 'native/g1_interface'), str(source), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
