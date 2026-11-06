"""Possession pipeline for More Than A Hack 2027, Phase 1.

Everything here reads the provided match assets through games.asset_paths() and writes a
frame,label CSV into evaluation/results/. Nothing prints raw tracking rows -- only aggregate
statistics -- so the pipeline can be developed without exporting match data anywhere.
"""
