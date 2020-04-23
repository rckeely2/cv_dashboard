#!/bin/sh

pip freeze > requirements.txt
cp requirements.txt container/
cp *.py container/
rm -r container/assets
cp -r assets container/
