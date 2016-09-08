import sys
import subprocess

brain_proc = None

def is_alive():
    global brain_proc
    # process was defined and it is still running (returncode is not defined yet)
    status = (brain_proc is not None) and (brain_proc.poll() is None)
    if not status:
        brain_proc = None
    return status

def run():
    if is_alive():
        print('brain is already running', file=sys.stderr)
        return
    # start brain and assign proc to brain_proc
    try:
        global brain_proc
        brain_proc = subprocess.Popen(['/home/stick/work/scm/pebe/riverbed-vision/webiface/blink.sh']) # FIXME: correct args to Popen
        print('brain started (PID %d)' % brain_proc.pid, file=sys.stderr)
    except Exception as e:
        print('brain start failed:', e, file=sys.stderr)

def kill():
    global brain_proc
    if is_alive():
        brain_proc.terminate() # this is for SIGTERM; use .kill() for SIGKILL
        print('brain killed (PID %d)' % brain_proc.pid, file=sys.stderr)
    brain_proc = None
