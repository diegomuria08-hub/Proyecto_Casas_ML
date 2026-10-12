"""Predice precios de casas usando el modelo guardado en este proyecto."""

from pathlib import Path
import math
import sys
import warnings


CARPETA_PROYECTO = Path(__file__).resolve().parent
RUTA_MODELO = CARPETA_PROYECTO / "modelo_casas.pkl"


def cargar_modelo():
    """Carga una sola vez el modelo que está junto a este script."""
    try:
        import joblib
    except ImportError as error:
        raise ImportError("Falta joblib. Instálalo con: py -m pip install joblib scikit-learn") from error

    if not RUTA_MODELO.is_file():
        raise FileNotFoundError(f"No se encontró el modelo: {RUTA_MODELO}")

    # Captura el aviso técnico largo de scikit-learn y lo muestra de forma breve.
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        modelo = joblib.load(RUTA_MODELO)

    for aviso in avisos:
        if "InconsistentVersionWarning" in aviso.category.__name__:
            print("Aviso: el modelo se guardó con otra versión de scikit-learn.")
            break

    if not callable(getattr(modelo, "predict", None)):
        raise TypeError("El archivo modelo_casas.pkl no contiene un modelo válido.")
    return modelo


def limpiar_pantalla():
    """Limpia CMD en Windows y ofrece una alternativa para otros sistemas."""
    import os

    os.system("cls" if os.name == "nt" else "clear")


def pedir_numero(mensaje, *, entero=False, minimo=0):
    while True:
        texto = input(mensaje).strip().replace(",", ".")
        try:
            valor = int(texto) if entero else float(texto)
            if not math.isfinite(valor):
                raise ValueError
            if valor < minimo:
                print(f"El valor debe ser mayor o igual que {minimo}.")
                continue
            return valor
        except ValueError:
            tipo = "entero" if entero else "numérico"
            print(f"Entrada inválida. Escribe un valor {tipo} válido.")


def pedir_otra_prediccion():
    while True:
        respuesta = input("\n¿Calcular otra casa? [S/N]: ").strip().lower()
        if respuesta in ("s", "si", "sí" "SI" "Si"):
            return True
        if respuesta in ("n", "no", "NO" "No"):
            return False
        print("Responde S para continuar o N para salir.")


def main():
    try:
        modelo = cargar_modelo()
    except Exception as error:
        print(f"No se pudo cargar el modelo: {error}", file=sys.stderr)
        print("Comprueba las dependencias con: py -m pip install joblib scikit-learn", file=sys.stderr)
        raise SystemExit(1) from error

    while True:
        limpiar_pantalla()
        print("=========================================")
        print(" Sistema de Predicción de Precios de Casas")
        print("=========================================")
        print("Modelo cargado. Introduce los datos de la casa.\n")

        try:
            metros = pedir_numero("Metros cuadrados de la casa: ", minimo=0.01)
            habitaciones = pedir_numero("Número de habitaciones: ", entero=True, minimo=1)
            antiguedad = pedir_numero("Antigüedad de la casa (años): ", minimo=0)

            # El orden coincide con el entrenamiento. Este modelo también acepta
            # una lista de valores; se oculta solo el aviso cosmético por nombres.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names.*",
                    category=UserWarning,
                )
                prediccion = float(modelo.predict([[metros, habitaciones, antiguedad]])[0])
            if not math.isfinite(prediccion):
                raise ValueError("El modelo devolvió un resultado no válido.")

            print("\n================ Predicción ================")
            print(f"Precio estimado: ${prediccion:,.2f} USD")
            print("============================================")

            if not pedir_otra_prediccion():
                break
        except (KeyboardInterrupt, EOFError):
            print("\nPrograma finalizado.")
            break
        except Exception as error:
            print(f"\nNo se pudo realizar la predicción: {error}")
            if not pedir_otra_prediccion():
                break

    print("\nPrograma finalizado.")


if __name__ == "__main__":
    main()
