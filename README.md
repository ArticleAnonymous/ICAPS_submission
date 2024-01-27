# Progressive State Space Disaggregation for Infinite Horizon Dynamic Programming

How to use the repository :
- pip install -r requirements.txt
- python main.py
- observe results in results/ directory (xlsx files with average runtime on 10 experience for each entry)

Models implemented here :
- Tandem Queue (Tournaire, Thomas, et al. "Factored reinforcement learning for auto-scaling in tandem queues." NOMS 2022-2022 IEEE/IFIP Network Operations and Management Symposium. IEEE, 2022.)
- Four Rooms (Hengst, Bernhard. "Hierarchical approaches." Reinforcement Learning: State-of-the-Art. Berlin, Heidelberg: Springer Berlin Heidelberg, 2012. 293-323.)
- Randomly generated MDPs (Archibald, T. W., K. I. M. McKinnon, and L. C. Thomas. "On the generation of markov decision processes." Journal of the Operational Research Society 46.3 (1995): 354-361.)
- Sutton tracks (Sutton, Barto, RL an Introduction Example 5.4 p.127)
- Barto tracks (Barto, Learning to Act Using Real-Time Dynamic Programming)
- Queue with Impatience (Jean-Marie, A., & Hyon, E. (2009). Scheduling in a queuing system with impatience and setup costs)
- Garnet MDPs (Bridging the gap between imitation learning and inverse reinforcement learning)

Remarks :
- You can observe first results in 5 minutes run but whole experimence takes several weeks.