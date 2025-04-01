from analyzer_v2 import analyze # type: ignore

import numpy as np
import matplotlib.pyplot as plt

def calculate_wavelength(pixel_pos,d_pixel_pos,D,d_D,g,d_g):
    #formula_1
    theta = np.arctan(pixel_pos/D)
    #Partielle Ableitungen
    dT_pixel_pos = (1/(1+(pixel_pos/D)**2))*(1/D)
    dT_D = (1/(1+(pixel_pos/D)**2))*(-pixel_pos/(D**2))
    d_theta = np.sqrt((dT_pixel_pos*d_pixel_pos)**2+(dT_D*d_D)**2)
    #formula_2
    lambda_ = np.sin(theta)*g
    #Partielle Ableitungen
    dl_g = np.sin(theta)
    dl_theta = np.cos(theta)*g

    d_lambda = np.sqrt((dl_g*d_g)**2+(dl_theta*d_theta)**2)

    return lambda_*1e9, d_lambda*1e9



def gauss_error_g(x, dx, D_cm, d_D_cm,lambda_, dlambda):
    sin_theta = np.sin(np.arctan(x / D_cm))
    g = (lambda_) / sin_theta

    # Partielle Ableitungen
    dg_x = lambda_ / (D_cm * (x**2 + D_cm**2))**0.5
    dg_D = -lambda_ * x / (D_cm**2 * (x**2 + D_cm**2))**0.5
    dg_lambda = 1/ sin_theta

    # Gaußsche Fehlerfortpflanzung
    dg = np.sqrt((dg_x * dx)**2 + (dg_D * d_D_cm)**2 + (dg_lambda * dlambda)**2)

    return g, dg

def get_D_screen(x,d_x,D_cm,d_D_cm,pixel_pos_ref,d_pixel_pos_ref):
    #formula
    D = D_cm*pixel_pos_ref/x
    
    #partial derivatives
    dD_d = pixel_pos_ref/x
    dD_pixel_pos_ref = D_cm/x
    dD_x = -1*D_cm*pixel_pos_ref/(x**2)

    d_D = np.sqrt((dD_d*d_D_cm)**2 + (dD_pixel_pos_ref*d_pixel_pos_ref)**2 + (dD_x*d_x)**2)
    return D,d_D

def berechne_abstand_zur_mitte(punkt_links, punkt_rechts, d_punkt_links, d_punkt_rechts):
    
    
    # Berechnung des Abstands zur Mitte für einen gegebenen Punkt (z. B. den Punkt der zugehörigen Farbe)
    abstand_mitte = (punkt_rechts - punkt_links) / 2  # Der Abstand zur Mitte der Farbe ist einfach der halbe Abstand zwischen links und rechts
    
    # Fehler des Abstands zur Mitte (Fehlerfortpflanzung der Positionen)
    d_abstand_mitte = 0.5*np.sqrt(d_punkt_links**2 + d_punkt_rechts**2)  # Fehler des Abstands zur Mitte
    
    return abstand_mitte, d_abstand_mitte

def berechne_abstand_zur_mitte_2(punkt,mitte , d_punkt, d_mitte):
    
    
    # Berechnung des Abstands zur Mitte für einen gegebenen Punkt (z. B. den Punkt der zugehörigen Farbe)
    abstand_mitte = abs(punkt - mitte)  # Der Abstand zur Mitte der Farbe ist einfach der halbe Abstand zwischen links und rechts
    
    # Fehler des Abstands zur Mitte (Fehlerfortpflanzung der Positionen)
    d_abstand_mitte = 0.5*np.sqrt(d_punkt**2 + d_mitte**2)  # Fehler des Abstands zur Mitte
    
    return abstand_mitte, d_abstand_mitte

def main(path=None,save_path=None):
  g,d_g = gauss_error_g(14.2e-2,0.1e-2,30e-2,0.1e-2,650e-9,10e-9)
  #print(g,d_g)
  peak_pos,d_peak_pos = berechne_abstand_zur_mitte(273,1447,10,10)
  
  #print(gauss_error_g(14e-2,0.1e-2,29.5e-2,0.1e-2,650e-9,10e-9))
  D,d_D = get_D_screen(14.2e-2,0.1e-2,30e-2,0.1e-2,peak_pos,d_peak_pos)
  
  pixel_error = 10
  
  #pixel_list = [(217,1519), (246,1548),(487,1237),(522,1245)]
  abstände = []
  values = []
  errors = []
  pixel_pos,d_pixel_pos = berechne_abstand_zur_mitte(465,1251,pixel_error,pixel_error)
  #print("wavelenght",calculate_wavelength(pixel_pos,d_pixel_pos,D,d_D,g,d_g))
  
  #for l,r in pixel_list:
  #   
  #   pixel_pos,d_pixel_pos = berechne_abstand_zur_mitte(l,r,pixel_error,pixel_error)
  #   abstände.append(pixel_pos)
  #   result_list.append(calculate_wavelength(pixel_pos,d_pixel_pos,D,d_D,g,d_g))
  
  #ran = int(input("enter center: "))
  ran = 850
  peak,clusters = analyze(path,save_path=f"{save_path}_graph.jpg")
  #print(peak)
  #print(clusters)
  for x in clusters:
      #print("mytext",x)
      
      for y in x:
        if y <1600:
          space,d_space = berechne_abstand_zur_mitte_2(y,peak,10,10)
          abstände.append(y)
          t,q = calculate_wavelength(space,d_space,D,d_D,g,d_g)
          
          errors.append(q)
          values.append(t)
  #print(abstände)
  #print(values)
  #print(errors)
  pixel_pos,d_pixel_pos = berechne_abstand_zur_mitte(465,1251,pixel_error,pixel_error)
  #print("wavelenght",calculate_wavelength(pixel_pos,d_pixel_pos,D,d_D,g,d_g))
  #
  #print("results list",result_list)
  
  
  values = list(values)
  errors = list(errors)
  
  #print(len(values),len(errors),len(abstände))
  #split data
  llist = [x for x in abstände if x < peak]
  #print(len(llist))
  x_values_1 = np.array(abstände[:len(llist):])  # X values from lowest to highest
  y_values_1 = np.array(values[:len(llist):])  # Corresponding Y values
  y_errors_1 = np.array(errors[:len(llist):])  # Error in Y
  #print(x_values_1)
  #print(y_values_1)
  x_values_2 = np.array(abstände[len(llist)::])  # X values from lowest to highest
  y_values_2 = np.array(values[len(llist)::])  # Corresponding Y values
  y_errors_2 = np.array(errors[len(llist)::])  # Error in Y
  
  # Create the plot
  fig, ax = plt.subplots()
  
  # Plot the main line
  ax.plot(x_values_1, y_values_1, linestyle='-', marker='o', color='blue', label="Test 1")
  
  # Add the error field
  ax.fill_between(x_values_1, y_values_1 - y_errors_1, y_values_1 + y_errors_1, color='blue', alpha=0.3, label="Error field")
  
  ax.plot(x_values_2, y_values_2, linestyle='-', marker='o', color='red', label="Test 2")
  ax.fill_between(x_values_2, y_values_2 - y_errors_2, y_values_2 + y_errors_2, color='red', alpha=0.3, label="Error field")
  
  
  # Labels and title
  ax.set_xlabel("Distance to 0 [pixel]")
  ax.set_ylabel("Wavelenght [nm]")
  ax.set_title("Wavelenghts of colors at a certain pixel")
  ax.legend()
  if save_path:
        plt.savefig(f"{save_path}_plot.jpg")
  plt.show(block=False)  # Show the plot without blocking execution
  plt.pause(1)           # Keep it open for 3 seconds
  plt.close() 
 
  
  