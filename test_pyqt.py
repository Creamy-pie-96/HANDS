import sys
try:
    from PyQt6.QtWidgets import QApplication, QLabel
    from PyQt6.QtCore import Qt
    print("PyQt6 imported successfully")
    app = QApplication(sys.argv)
    label = QLabel("Hello PyQt6")
    
    # Test the attribute
    try:
        # Note: In PyQt6, attributes are in Qt.WidgetAttribute
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        print("Set WA_TransparentForMouseEvents successfully")
    except AttributeError:
        print("AttributeError: WA_TransparentForMouseEvents not found")
        
    label.show()
    print("PyQt6 app initialized")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
