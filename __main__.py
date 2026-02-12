import mlx.core as mx

# 1. Check the default device (Should be GPU)
# Vérifier le device par défaut (Devrait être GPU)
print(f"Default device: {mx.default_device()}")

# 2. Create an array (It lives in Unified Memory)
# Créer un tableau (Il vit dans la mémoire unifiée)
a = mx.array([1, 2, 3])
b = mx.array([1, 2, 3])

# 3. Perform an operation (This will run on the GPU implicitly)
# Faire une opération (Cela s'exécutera sur le GPU implicitement)
c = a + b
mx.eval(c) # Force computation (MLX is lazy!) / Force le calcul

print(f"Result: {c}")