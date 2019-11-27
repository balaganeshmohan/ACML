import gym 
import numpy as np
import matplotlib.pyplot as plt
slow = False


env = gym.make("MountainCar-v0")

#parameters
velocity = 20
position = 20
actions = 3
alpha = 0.1
gamma = 0.9
epsilon = 0



Q_value = np.zeros((position,velocity,actions))
Alpha = np.zeros((position,velocity,actions))
Visit = np.zeros((position,velocity,actions))


#Function for defining actions
def Action(pos,speed):
   
    prob = np.random.uniform(0,1)
    if(prob>epsilon):
        a = np.argmax(Q_value[pos,speed]);
    else:
        a = env.action_space.sample();
    
    return a;


#Function for output heatmap

def heatmap2d(arr: np.ndarray):
    # helper function for visualization
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
	#plt.save("C:dke\acml\4\result.png")

	

#Get maximum Q value	
def MaxQ():
    
    maxQ_value = np.zeros((position,velocity))
    for i in range(position):
        for j in range(velocity):
            maxQ_value[i,j] = np.max(Q_value[i,j]);
            np.save("Q-value",maxQ_value)
            
    
	
	
#Q_learning learning algorithm   
def Q_learning():
    
    for i in range(10000):
        if(i>9995):
            slow = True;
        else:
            slow = False;
        observation = env.reset()
        done = False
        timesteps = 0
        maxPosition = env.observation_space.high[0]
        minPosition = env.observation_space.low[0]
        maxSpeed = env.observation_space.high[1]
        minSpeed = env.observation_space.low[1]
    
       

        while not done:
            if slow: env.render()
            
        
            pos = np.math.floor(((observation[0]-minPosition)/(maxPosition-minPosition)) * position);
            speed = np.math.floor(((observation[1]-minSpeed)/(maxSpeed-minSpeed)) * velocity);
        
            action = Action(pos,speed) 

            observation, reward, done, info = env.step(action)
        
            npos = np.math.floor(((observation[0]-minPosition)/(maxPosition-minPosition)) * position);
            nspeed = np.math.floor(((observation[1]-minSpeed)/(maxSpeed-minSpeed)) * velocity);
            Q_value[pos,speed,action]  = Q_value[pos,speed,action] + (alpha * ((reward + gamma * np.max(Q_value[npos,nspeed]))-Q_value[pos,speed,action]))
           
            
            
            timesteps+=1
            if slow: print(observation)
            if slow: print(reward)
            if slow: print(done)
        print("Episode finished after ", timesteps, "timesteps.")
    if(slow==True):
        env.render(close=False)
    
            
    
    
#main   
if __name__ == "__main__":
    Q_learning();
    MaxQ();
    
heatmap2d(MaxQ_value)
fig = plt.figure()



    
            
        
    
    
