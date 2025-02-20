# notes

- [documentation for JAX on ginsburg](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62141879/Ginsburg+-+Job+Examples#Getting-Python-libraries-JAX-and-JAXLIB-to-work-with-GPUs)

## Dated Thoughts

### February 3rd
- A couple thoughts. First, I wonder if I can increase the accuracy/decrease the loss of my khcloudnet dataset if I look at the following: (1) removing annotations below a certain cutoff (could be related to the weird thing of having annotations with values >10 <90 for thresholding if I'm not mistaken; either way, annotations that are too small are an issue), (2) seeing if whispy clouds are causing problems (write exclusion script to remove them to check?), (3) in general reviewing annotations to see which ones to remove from set, and (4) maybe edge width of 2 for supervisely annotations is conflicting with threshold ones. Second, I could also justify the lower accuracy as result of not having multispectral (I should still compare to other potential methods for baseline, like random forest). Third, I think I'm in good spot for three papers still: KHCloudNet, KHSelect, and TransformerTrainer

- Need to get Jax working tested for NASA. 

### February 10th

- Regarding ginsburg, trying to clear out directories and find where python files are stored. My partition is currently 2.8 gb total. the `tjk2147/.local` directory is 2.2 GB alone. (src is ~574 MB). 

- Okay. From investigation it appears that I can (and did) clear the `tjk2147/.local` directory because from execute of the commands `module load anaconda/3-2023.09`, `which python`, `python -c "import sys; print(sys.path)"` and `conda list | grep torch` in `script.sh`, it appears that the python related things (interpreter, library, etc.) being used while I am in the slurm run are coming from outside my .local directory. 

- I just made my own conda distribution within slurm run (previously I installed conda in command line and work with it outside slurm, but that was a waste of space) and installed all my libraries, my total storage use is about 8.9 GB now (max is 50 GB), so it's all good. 


### NASA Meeting 2/10 

-> good: <1% (no cloud cover), maybe is (~10%, ideally localized). huge range in bad, useless is fully covered. 