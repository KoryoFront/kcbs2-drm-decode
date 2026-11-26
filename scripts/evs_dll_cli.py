"""Invoke the locally built EVS codec DLL in its own process."""
import ctypes
import os
import sys


def main(args):
    codec = ctypes.CDLL(os.path.abspath(args[0]))
    codec.evs_main.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    encoded = [os.fsencode(arg) for arg in args]
    argv = (ctypes.c_char_p * (len(encoded) + 1))(*encoded, None)
    return codec.evs_main(len(encoded), argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
