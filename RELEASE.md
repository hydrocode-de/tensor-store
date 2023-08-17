# TensOrage

TensOrage is a Python package that can be used to store tensor-like data into a supabase backend.
It works with hosted and self-hosted backends. It is focused to utilize numpy-like slicing of 
tensors which have a first axis, that is way larger than the other axes.
This way, models can be trained iteratively with data, that does not fit into memory.
TensOrage does also use supabase authentication to implement multi-user and can even define
quotas.

Install instructions, quickstart and a note about Supabase will follow.