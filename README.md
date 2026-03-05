# Suma de Riemann en Python (Vercel)

Aplicación web en Flask + SymPy para:

- Recibir una función `f(x)` ingresada por el usuario.
- Calcular suma de Riemann simbólica con `n` subintervalos.
- Evaluar la suma para un `n` específico.
- Soportar métodos: izquierda, derecha, centro y trapezoidal.
- Mostrar aproximación numérica opcional.
- Dibujar y sombrear el área de la suma de Riemann.
- Aceptar expresiones como `x^2`, `e^x`, `sen(x)`, `ln(x)`, `log10(x)`,
  funciones trigonométricas e hiperbólicas, y parámetros simbólicos (`a`, `b`, etc.).

## Ejecutar localmente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python api/index.py
```

Abrir `http://127.0.0.1:5000`.

## Despliegue en Vercel

1. Importa este repositorio en Vercel.
2. Vercel detectará `vercel.json` y usará `@vercel/python`.
3. Asegúrate de que `requirements.txt` está en la raíz.
