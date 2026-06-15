"""Pre-train long + short models for every symbol in config.SYMBOLS.

Run once locally before first Railway deploy so the bot starts instantly
(no ~10min/pair lazy training on container boot). Skips symbols whose
model file already exists — safe to re-run after interruption.
"""

import os
import sys
import time
import traceback

import config
from model_trainer import (
    model_file_for,
    short_model_file_for,
    train_and_save,
    train_and_save_short,
)


def main():
    symbols = config.SYMBOLS
    total = len(symbols) * (2 if config.ENABLE_SHORTING else 1)
    done = 0

    for sym in symbols:
        mf = model_file_for(sym)
        if os.path.exists(mf):
            print(f"[train_all] {sym} long model exists, skip")
        else:
            t0 = time.time()
            try:
                train_and_save(sym, mf)
                print(f"[train_all] {sym} LONG done in {time.time()-t0:.0f}s")
            except Exception as e:
                print(f"[train_all] {sym} LONG FAILED: {e}")
                traceback.print_exc()
        done += 1
        print(f"[train_all] progress {done}/{total}")

        if config.ENABLE_SHORTING:
            smf = short_model_file_for(sym)
            if os.path.exists(smf):
                print(f"[train_all] {sym} short model exists, skip")
            else:
                t0 = time.time()
                try:
                    train_and_save_short(sym, smf)
                    print(f"[train_all] {sym} SHORT done in {time.time()-t0:.0f}s")
                except Exception as e:
                    print(f"[train_all] {sym} SHORT FAILED: {e}")
                    traceback.print_exc()
            done += 1
            print(f"[train_all] progress {done}/{total}")

    print("[train_all] ALL DONE")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
