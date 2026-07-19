.PHONY: smoke manifest paper paper-anonymous

smoke:
	python3 code/replication_smoke.py
	python3 code/build_artifact_manifest.py --check

manifest:
	python3 code/build_artifact_manifest.py --write

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

paper-anonymous:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main_anonymous.tex
