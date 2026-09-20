# CUDA/NCCL syntax stubs

Just enough declarations to let a host C++ compiler type-check `.cu` sources
on a machine with no CUDA toolkit. **Not an implementation** — every function
is declared and never defined, so the check is `-fsyntax-only`; nothing here
links or runs, and nothing here can be used to fake a GPU result.

Why it exists: `nccl_main.cu` produced every published GPU number in this
repository, and it has repeatedly been edited on a laptop that cannot compile
it. That is how a `printf` shipped two JSON fields as `(null)` and dropped a
configuration out of a benchmark matrix an hour into a paid run.

What it does and does not catch:

* **catches** — type errors, wrong argument counts, undeclared names, and
  `printf` format/argument mismatches, which is the defect class that has
  actually bitten this project. The build's own `-Wformat=2 -Werror=format`
  are applied here too.
* **does not catch** — launch configuration, device-side semantics, memory
  errors, or anything about NCCL's real behaviour. `__global__` bodies are
  compiled as ordinary host functions with shimmed warp intrinsics, so they
  type-check but do not mean anything.

Passing this is not evidence a kernel is correct. It is evidence the file
would survive `nvcc` far enough to find out.
