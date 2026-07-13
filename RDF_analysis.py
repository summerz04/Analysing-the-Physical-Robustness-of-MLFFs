from CRISP.data_analysis.prdf import analyze_rdf 

files = ['xtb_test.traj']

# setting rdf analysis arguments 
traj_path = './xtb_test.traj'
rmax = 8.0
nbins = 50
frame_skip=50
output_dir = 'rdfs'
output_filename = 'test.pkl'
use_prdf = False
atomic_indices = None

print('Beginning RDF analysis')
for file in files: 
    print(f'Reading {file}')
    test_rdf = analyze_rdf(
        use_prdf=False,
        rmax=rmax,
        traj_path=traj_path,
        nbins=nbins,
        frame_skip=frame_skip,
        output_filename=output_filename,
        atomic_indices=None,
        create_plots=True
    )
