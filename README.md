# Newbie-Teacache-ComfyUI

Most of the code here was written with the help of multiple coding assistants (mainly Claude and Gemini).
These nodes contain a patched loader for using TeaCache with NewBie and a Coefficient Calculator that I used to get a okayish coefficient.

Just place the "TeaCache (Newbie)" on your workflow and hook up the model to it. By default it uses the best coefficient I could find ([0, 0, 0, 4.11423217, 0.36885889]) and a speed-focused l1 value. You can set it between 0.3 (slowest, best quality) and 0.8 (fastest, lower quality).

The Coefficient Calculator is more of a testing node (I don't even know if it's optimal or not), mess with it if you want to.
