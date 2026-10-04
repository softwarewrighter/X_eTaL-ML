# the 1.58-bit network at the command line, recorded with VHS: just record ternary-net
# renders it to demos/ternary-net/recording.webp (scripts/record.sh).
Output work/record/ternary-net.gif
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

Type "# X_eTaL-ML: the 1.58-bit network, run by the vendored X_eTaL" Enter
Sleep 800ms
Type "just xetal-version" Enter
Sleep 2.5s
Type "clear" Enter
Type "# One classifier, its weights stored as FP32, FP16, INT8 and ternary -1 0 +1; each statement, then its result" Enter
Sleep 800ms
Type "just tour ternary-net" Enter
Sleep 35s
