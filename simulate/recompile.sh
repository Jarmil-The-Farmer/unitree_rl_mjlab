#!/bin/bash

# 1. Získání absolutní cesty k adresáři, kde se nachází tento skript
# Díky tomu bude "build" vždy ve stejné složce jako skript, i když ho spustíte odjinud.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# 2. Definice cesty k build složce (v uvozovkách kvůli mezerám)
BUILD_PATH="$SCRIPT_DIR/build"

echo "Začínám rekompilaci v: $BUILD_PATH"

# 3. Smazání starého buildu a vytvoření nového
# -p u mkdir zajistí, že skript neselže, pokud by složka už existovala (pro jistotu)
rm -rf "$BUILD_PATH"
mkdir -p "$BUILD_PATH"

# 4. Přesun do build složky
cd "$BUILD_PATH" || exit 1

# 5. Konfigurace projektu pomocí CMake
# ".." odkazuje na složku nad buildem (kde by měl být CMakeLists.txt)
cmake ..

# 6. Kompilace pomocí všech dostupných jader
# nproc zjistí počet vláken procesoru
echo "Spouštím make s $(nproc) jádry..."
make -j$(nproc)

echo "Hotovo!"