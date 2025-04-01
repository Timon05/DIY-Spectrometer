import os
from error_v3 import main
for files in os.listdir("./Downloads/photos/originals"):
  print(files)
  
  main(path=f"C:\\Users\\timon\\Downloads\\photos\\originals\\{files}",save_path=f"C:\\Users\\timon\\Downloads\\photos\\{files}")

 