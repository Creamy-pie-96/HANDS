
import sys
import platform
import json
import os
from pathlib import Path

class OSHandler:
    """
    Handles OS detection and configuration persistence.
    Designed to make the application cross-platform by identifying the running OS
    and saving it to the configuration file for faster future lookups.
    """
    
    @staticmethod
    def detect_os() -> str:
        """
        Detect the current operating system.
        
        Returns:
            str: 'linux', 'windows', 'darwin' (macOS), or 'unknown'
        """
        system = platform.system().lower()
        if system == 'linux':
            return 'linux'
        elif system == 'windows':
            return 'windows'
        elif system == 'darwin':
            return 'darwin'
        return 'unknown'
    
    @staticmethod
    def get_os(config=None) -> str:
        """
        Get the current OS, preferably from config, otherwise by detection.
        If detected fresh, it updates the config.
        
        Args:
            config: Config instance or None. If None, we just detect.
            
        Returns:
            str: 'linux', 'windows', 'darwin', or 'unknown'
        """
        # Try to read from config if provided
        if config:
            stored_os = config.get('system', 'os_type', default=None)
            if stored_os:
                return stored_os
        
        # Detect
        detected_os = OSHandler.detect_os()
        
        # Save if config is available
        if config:
            # We need to update the config properly
            # Assuming config has a way to set values or we need to edit file directly?
            # The Config class in this project seems to have .get methods, let's assume it might not have .set 
            # or we should use a safe way to update config.
            # If the Config object is read-only or we shouldn't modify it at runtime like this,
            # we might just return the detected OS.
            #
            # However, the user requirement is: "make the os saved in the config.json and read from it"
            # So we should try to save it.
            try:
                # Basic attempt to save to file if Config doesn't have a 'set' method exposed here easily
                # But safer to just rely on detection if we can't easily write back through the object.
                # Let's check config path from the object if possible.
                if hasattr(config, '_config_path'):
                    OSHandler._update_config_file(config._config_path, detected_os)
                elif hasattr(config, 'config_path'):
                    OSHandler._update_config_file(config.config_path, detected_os)
            except Exception as e:
                print(f"⚠ Failed to save OS to config: {e}")
                
        return detected_os

    @staticmethod
    def _update_config_file(config_path: str, os_name: str):
        """
        Helper to safely update the config json file with the OS type.
        """
        try:
            if not os.path.exists(config_path):
                return
                
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            # Ensure 'system' section exists
            if 'system' not in data:
                data['system'] = {}
            
            # Check if it's already there to avoid unnecessary writes
            current_val = data['system'].get('os_type')
            
            # The config format often uses [value, description] lists.
            # Check if we should convert to that format.
            # Based on the config.json seen earlier, sections like "gestures_enabled" use lists.
            # "system_control" -> "cursor" -> "smoothing_factor": [0.6, "desc"]
            # Let's try to match that style if 'system' section is empty or follows that style.
            
            # If 'system' is a new dict we just created, we can define the structure
            if isinstance(current_val, list):
                if current_val[0] == os_name:
                    return # No change needed
                data['system']['os_type'][0] = os_name
            else:
                # Just set the value directly if it's not a list, OR create a new list entry
                # Best to safeguard and just write the value if we aren't sure, 
                # but to be consistent with the project style:
                data['system']['os_type'] = [os_name, "Detected Operating System (linux/windows/darwin)"]
            
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠ Error updating config file: {e}")
