# the MoE routing microscope at the command line, recorded with VHS: just record moe-router
# renders it to demos/moe-router/recording.webp (scripts/record.sh).
Output work/record/moe-router.gif
Require just
Set Shell "bash"
Set FontSize 15
Set Width 1280
Set Height 820
Set Padding 24
Set TypingSpeed 30ms
Set Theme "Dracula"

Hide
Type "export PS1='$ '; clear" Enter
Sleep 500ms
Show

Type "# X_eTaL-ML: the MoE routing microscope, run by the vendored X_eTaL" Enter
Sleep 800ms
Type "just xetal-version" Enter
Sleep 2.5s
Type "clear" Enter
Type "# Tokens scored against 16 experts by one matrix product, softmax, top-2 by masks; each statement, then its result" Enter
Sleep 800ms
Type "just tour moe-router" Enter
Sleep 35s
