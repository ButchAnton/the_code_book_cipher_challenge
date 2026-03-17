# Cipher Challenge Solutions (mostly) in Python

This is an attempt at re-solving the Cipher Challenge
presented in the original edition of Simon Singh's
_The Code Book_.

I decided to go back and solve the challenge using
Python. My original solutions, lost to some dead hard
drive somewhere, were written in C.  

By and large, in this attempt, the early stages are
solved using Python.  However, due to performance
issues, later solutions or parts of solutions are
written in C/C++.

Also, please note that stage 9 is solved using an
NVIDIA GPU.  This stage must be solved using CUDA
under Windows.  Sorry about that.  Everything else
should be able to run pretty much anywhere, though
there may be some Windows-specific crap some places
that snuck in.  I ended up doing a lot of the work
in later stages using Windows, because, well, I
needed more umph than my MacBook Pro could give me.

I have tried to automate as many of the solutions as
possible. My previous solutions required a lot of
hand analysis, which I have tried to move into the
code.
