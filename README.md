# Progressive State Space Disaggregation for Infinite Horizon Dynamic Programming

How to use the repository in few steps :
- git clone https://github.com/ArticleAnonymous/ICAPSagg.git
- cd ICAPSagg
- pip install -r requirements.txt
- python main.py
- observe results in results/discounted_avg_runtime_ICAPS_agg_0.xlsx

Models implemented here :
- barto_track : Barto, Andrew G., Steven J. Bradtke, and Satinder P. Singh. "Learning to act using real-time dynamic programming." Artificial intelligence 72.1-2 (1995): 81-138.
- rooms : Hengst, Bernhard. "Hierarchical approaches." Reinforcement Learning: State-of-the-Art. Berlin, Heidelberg: Springer Berlin Heidelberg, 2012. 293-323.
- garnet : Bhatnagar, Shalabh, et al. "Natural actor–critic algorithms." Automatica 45.11 (2009): 2471-2482.
- tandem_queue : Tournaire, Thomas, et al. "Factored reinforcement learning for auto-scaling in tandem queues." NOMS 2022-2022 IEEE/IFIP Network Operations and Management Symposium. IEEE, 2022.
- sutton_track : Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018., Example 5.4 p.127
- weakly-couplet garnets

Remarks :
- Whole experimence takes several days as runtime is measured on 10 experience for each model and each solver
- discount = 0.999, epsilon = 1e-3
- All solvers are implemented in the "solvers" folder, and all models in the "models" solver
