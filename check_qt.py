from PyQt5.QtCore import Qt
print("WA_TransparentForInput" in dir(Qt))
print([a for a in dir(Qt) if "Transparent" in a])
