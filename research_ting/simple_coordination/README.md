# Simple Coordination

This is a simple coordination task that extends on the GRU network and environment made in TFE4580 - "Anti-Jamming using Deep Q-learning".

The program in this directory extends on this to test the idea of cooperative multi-agent reinforcement learning. The agents are trained using Independent Deep Double Q-learning Networks (IDDQN) and are tested both for 10 channels and 100 channels. To maintain a successful communication link, the Rx must select the same band or one of the neighboring bands as the Tx. The Rx agent gets a reward if it is able to decode the message sent by the Tx agent. The Tx gets a reward if it is able to detect the ACK sent by the Rx.