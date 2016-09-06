import subprocess

brain_proc = None

def is_alive():
    # process was defined and it is still running (returncode is not defined yet)
    status = (brain_proc is not None) and (brain_proc.poll() is None)
    if not status:
        brain_proc = None
    return status

def run():
    if is_alive():
        print('brain is already running')
        return
    # start brain and assign proc to brain_proc
    brain_proc = subprocess.Popen('../brain/brain.py') # FIXME: correct args to Popen

def kill():
    if is_alive():
        brain_proc.terminate() # this is for SIGTERM; use .kill() for SIGKILL
    brain_proc = None
