# ActiveSelection

The project is maven project. The intellij is used as an ide. It requires JDK11 and Scala 2.13.8 in order to compile. Other than these two environments, Intel Oneapi 2023.2 may be required and LD_LIBRARY must be specified to locate the correct intelone api bindings. 

In ActiveSelection library several active selection approaches are implemented to extract representative sentences from a large text corpus. The code contains two experimental stages. In the first stage the dataset is constructed. 
In this stage, an active selection method uses an Z-score based online moving scoring average to obtain a score and decide on the informativeness of the target sentence. The sentences having sginificant scores are selected from this set and saved to a text file line by line.
In the second stage, a deep learning model is used to extract word embeddings from the selected sentences. Later, these embeddings are used to evaluate the quality of the embeddings. The active selection methods are compared with the same set of embedding extraction method and intrinsic/extrinsic evaluation dataset.
The evaluations are saved to resources/results folder along with the parameters. The parameters are used for hasing the result file. So, for each set of parameters such as extraction model, embedding vector length, and selection method, there should be a unique result file.   

ExperimentSPL is the main entry point for the first stage of the program. In ExperimentalSPL the parameters relevant to the selected active selection methods are used to create a dataset consists of selected sentences. 
ExperimentSPL requires a large line by line text corpus and a large word embedding binary. Before running ExperimentSPL both files must be placed within paths resources/text/sentences/sentences-tr.txt and resources/binary/huewei.bin respectively. A general overview of this processing pipeline is schematized as follows.

![ExperimentSPL processing steps](https://github.com/volkanagun/ActiveSelection/blob/master/pipeline.jpg?raw=true)

An important part of this processing pipeline is the construction of the static vocabulary corpus. In this step, the dictionary of the target evaluation tests are used to filter the sentences. In this respect, each target word is used to sample the corpus sentences randomly so that each word will have equal number of sentences. Note that the sampled sentences must be also distinct.     

In this procesing pipeline, before the selection process the dictionaries of the target evaluation datasets must be constructed. 
