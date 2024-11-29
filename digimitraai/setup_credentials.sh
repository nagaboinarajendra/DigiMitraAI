#!/bin/bash

# Print start message
echo "Starting credentials setup..."

# Set project directory
PROJECT_DIR="/workspaces/DigiMitraAI/digimitraai"

# Create credentials directory if it doesn't exist
mkdir -p "$PROJECT_DIR/credentials"
echo "Created credentials directory"

# If using Codespaces secrets (GOOGLE_CREDENTIALS environment variable)
if [ -n "$GOOGLE_CREDENTIALS" ]; then
    echo "Found Google Cloud credentials in Codespaces secrets"
    
    # Save credentials to file
    echo "$GOOGLE_CREDENTIALS" > "$PROJECT_DIR/credentials/google-cloud-credentials.json"
    
    # Set proper permissions
    chmod 600 "$PROJECT_DIR/credentials/google-cloud-credentials.json"
    
    echo "Credentials file created and secured"
    
    # Set environment variable
    export GOOGLE_APPLICATION_CREDENTIALS="$PROJECT_DIR/credentials/google-cloud-credentials.json"
    
    # Add to bashrc for persistence
    echo "export GOOGLE_APPLICATION_CREDENTIALS=\"$PROJECT_DIR/credentials/google-cloud-credentials.json\"" >> ~/.bashrc
    
    echo "Environment variable set successfully"
else
    echo "WARNING: No credentials found in Codespaces secrets"
    echo "Please either:"
    echo "1. Add GOOGLE_CREDENTIALS to your Codespaces secrets, or"
    echo "2. Manually upload your credentials file to $PROJECT_DIR/credentials/google-cloud-credentials.json"
fi

# Verify file exists
if [ -f "$PROJECT_DIR/credentials/google-cloud-credentials.json" ]; then
    echo "✅ Credentials file exists"
    
    # Check permissions
    PERMS=$(stat -c %a "$PROJECT_DIR/credentials/google-cloud-credentials.json")
    if [ "$PERMS" = "600" ]; then
        echo "✅ Credentials file permissions are correct"
    else
        echo "⚠️ Warning: Credentials file permissions are $PERMS, fixing..."
        chmod 600 "$PROJECT_DIR/credentials/google-cloud-credentials.json"
    fi
else
    echo "❌ Credentials file not found"
fi

echo "Credentials setup completed"