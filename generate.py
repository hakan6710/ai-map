#Script zum Erstellen vom Jupyter Book
#TOC-File wird generiert und danach jupyter build aufgerufen

from pickle import TRUE
import subprocess

toc_data=subprocess.run("jupyter-book toc from-project -f jb-article ./docs --extension .ipynb --extension .md",shell=True,text=True, capture_output=True)

toc_data_cleaned=toc_data.stdout.replace("ai_map","docs/ai_map")
toc_data_cleaned=toc_data_cleaned.replace("start","docs/start")

f = open("_toc.yml", "w")
f.write(toc_data_cleaned)
f.close()

subprocess.run("jupyter-book build --all .",shell=True)
