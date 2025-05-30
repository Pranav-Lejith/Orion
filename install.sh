#!/bin/bash

echo "Installing Orion Language..."

# Define installation directory
ORION_DIR="/usr/local/orion"

# Create directory if it doesn't exist
sudo mkdir -p "$ORION_DIR"

# Copy the interpreter to the installation directory
sudo cp "$(dirname "$0")/interpreter.py" "$ORION_DIR/"

# Create a script to run the interpreter
sudo bash -c "cat > $ORION_DIR/orion" <<EOL
#!/bin/bash
python3 "$ORION_DIR/interpreter.py" "\$@"
EOL

# Make it executable
sudo chmod +x "$ORION_DIR/orion"

# Add /usr/local/orion to PATH if it's not already
if [[ ":$PATH:" != *":$ORION_DIR:"* ]]; then
  SHELL_CONFIG="$HOME/.bash_profile"
  [ -f "$HOME/.zshrc" ] && SHELL_CONFIG="$HOME/.zshrc"
  echo "export PATH=\"\$PATH:$ORION_DIR\"" >> "$SHELL_CONFIG"
  echo "Added Orion to PATH in $SHELL_CONFIG. Please restart your terminal or run: source $SHELL_CONFIG"
fi

echo "Installation complete. You can now use 'orion run <filename.pl>' or 'orion shell' in terminal."
