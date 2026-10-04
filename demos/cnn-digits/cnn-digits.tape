# the tiny CNN at the command line, recorded with VHS: just record cnn-digits
# renders it to demos/cnn-digits/recording.webp (scripts/record.sh).
Output work/record/cnn-digits.gif
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

Type "# X_eTaL-ML: the tiny CNN, run by the vendored X_eTaL" Enter
Sleep 800ms
Type "just xetal-version" Enter
Sleep 2.5s
Type "clear" Enter
Type "# A convolutional network reads MNIST digits; weights trained offline; each statement, then its result" Enter
Sleep 800ms
Type "just tour cnn-digits" Enter
Sleep 35s
