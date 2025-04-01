import os
from error_v3 import main
for files in os.listdir("./thisfolder/photos/originals"):
  print(files)
  
  main(path=f"path\\to\\folder{files}",save_path=f"path\\to\\folder\\photos{files}")

 
