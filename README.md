<div align="center">
  <h1 align="center"> DEM-Vibropacking-Pearson-Distributions </h1>
</div>

This repository provides all the code files used within the paper "Effect of standard deviation, skew and kurtosis on the packing density of continuous particle size distributions vibrated in one dimension".

## Code Run Order

The run order of both the shell launch scripts, and the file it launches are outlined below. All drop down directories below can be run without any dependency on files in the other drop down directories. 

<details markdown="1"><summary><h3><a href="./Pearson_PDF_python">Pearson_PDF_python</a></h3></summary>
  
  1\) [pearson_pdf.py](./Pearson_PDF_python/pearson_pdf.py) <br />
  
</details>

<details markdown="1"><summary><h3><a href="./studied_distributions">studied_distributions</a></h3></summary>
  
  1\) [kurtosis_distribution_generator.py](./studied_distributions/kurtosis_distribution_generator.py)
  <br />
  1\) [skew_distribution_generator.py](./studied_distributions/skew_distribution_generator.py)
  <br />
  1\) [standard_deviation_distribution_generator.py](./studied_distributions/standard_deviation_distribution_generator.py)
  <br />
  
</details>

<details markdown="1"><summary><h3><a href="./monodisperse_study">monodisperse_study</a></h3></summary>
  
  1\)  [launch_distribution_generator.sh](./monodisperse_study/launch_distribution_generator.sh) &#8594; [distribution_generator.py](./monodisperse_study/distribution_generator.py)
  <br />
  2\) [launch_final_packing.sh](./monodisperse_study/launch_final_packing.sh) &#8594; [final_packing.py](./monodisperse_study/final_packing.py)
  <br />

</details>

<details markdown="1"><summary><h3><a href="./standard_deviation_study">standard_deviation_study</a></h3></summary>
  
  1\)  [launch_distribution_generator.sh](./standard_deviation_study/launch_distribution_generator.sh) &#8594; [distribution_generator.py](./standard_deviation_study/distribution_generator.py)
  <br />
  2\) [launch_final_packing.sh](./standard_deviation_study/launch_final_packing.sh) &#8594; [final_packing.py](./standard_deviation_study/final_packing.py)
  <br />
  3\) [standard_deviation_results_graph.py](./standard_deviation_study/standard_deviation_results_graph.py)
  <br />

</details>

<details markdown="1"><summary><h3><a href="./skew_study">skew_study</a></h3></summary>
  
  1\)  [launch_distribution_generator.sh](./skew_study/launch_distribution_generator.sh) &#8594; [distribution_generator.py](./skew_study/distribution_generator.py)
  <br />
  2\) [launch_final_packing.sh](./skew_study/launch_final_packing.sh) &#8594; [final_packing.py](./skew_study/final_packing.py)
  <br />
  3\) [skew_results_graph.py](./skew_study/skew_results_graph.py)
  <br />

</details>

</details>

<details markdown="1"><summary><h3><a href="./kurtosis_study">kurtosis_study</a></h3></summary>
  
  1\)  [launch_distribution_generator.sh](./kurtosis_study/launch_distribution_generator.sh) &#8594; [distribution_generator.py](./kurtosis_study/distribution_generator.py)
  <br />
  2\) [launch_final_packing.sh](./kurtosis_study/launch_final_packing.sh) &#8594; [final_packing.py](./kurtosis_study/final_packing.py)
  <br />
  3\) [kurtosis_results_graph.py](./kurtosis_study/kurtosis_results_graph.py)
  <br />

