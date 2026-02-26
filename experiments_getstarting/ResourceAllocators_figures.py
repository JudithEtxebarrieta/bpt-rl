'''
Environments:
-InvertedPendulum
-InvertedDoublependulum

Methods:
-RS: random search
-SH: successive halving
-PSH soft/original: progressive SH using soft or original stability of rankings

Experiments:
- Learning-curves: no son monotono crecientes porque estoy considerando la ultima politica visitada, en lugar de la mejor.
- Trade-off curves: aplicar estos metodos solo tiene sentido si el tiempo total disponible nos permite hacer multiples ejecuciones hasta convergencia,
este es el caso en que es coherente plantear distribuir los recursos entre diferentes procesos. Se observa que los metodos basados en SH son mucho mas
estables al numero de procesos considerados que el RS.
- Test reward and train time comparison: para tiempos de ejecucion coherentes, los basados en SH superan a RS. Esto indica que la distribucion no 
uniforme de los recursos entre los procesos es mas eficiente. Por otra parte, entre los diferentes SH el test reward obtenido es parecido, pero el
tiempo consumido para llegar a ese score es menor para los PSH.

'''
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math
import itertools

#--------------------------------------------------------------------------------------------------
# Funciones generales
#--------------------------------------------------------------------------------------------------
def bootstrap_mean_and_confidence_interval(data,bootstrap_iterations=1000):
    '''
    The 95% confidence interval of a given data sample is calculated.

    Parameters
    ==========
    data (list): Data on which the range between percentiles will be calculated.
    bootstrap_iterations (int): Number of subsamples of data to be considered to calculate the percentiles of their means. 

    Return
    ======
    The mean of the original data together with the percentiles of the means obtained from the subsampling of the data. 
    '''
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

def concat_df(list_seeds,env_name, path):
    df = pd.read_csv(path+'df_'+env_name+'_seed'+str(list_seeds[0])+'.csv')
    list_seeds.pop(0)
    for seed in list_seeds:
        df_new=pd.read_csv(path+'df_'+env_name+'_seed'+str(seed)+'.csv')
        df = pd.concat([df, df_new], ignore_index=True)

    return df

def from_matrix_to_ycoordinates(matrix):

    list_y_mean=[]
    list_y_q05=[]
    list_y_q95=[]
        
    for i in range(matrix.shape[0]):
        y_mean,y_q05,y_q95=bootstrap_mean_and_confidence_interval(matrix[i,:])
        list_y_mean.append(y_mean)
        list_y_q05.append(y_q05)
        list_y_q95.append(y_q95)

    return [list_y_mean,list_y_q05,list_y_q95]

def n_subsets_of_fixed_len(set,n_subsets,len_subsets):
    list_sub=[]
    for _ in range(n_subsets):
        list_sub.append(random.sample(set,len_subsets))

    return list_sub

def sort_list_according_to_argsort_list(list_to_sort,argsort_list):
    new_list=[]
    for i in range(len(list_to_sort)):
        new_list.append(list_to_sort[argsort_list[i]])
    return new_list

def possible_B_values_for_eta(max_B,eta):

    def compute_s(B, eta):
        input_s = 0 
        output_s=None
        while output_s!=input_s:
            output_s = math.floor(math.log(B / (input_s + 1), eta))
            input_s+=1
            if output_s<0:
                return False
        return True

    list_B=[]
    for B in range(1,max_B+1):
        if compute_s(B,eta):
            list_B.append(B)
    return list_B


#--------------------------------------------------------------------------------------------------
#Funcion para dibujar graficas de curvas trade off
#--------------------------------------------------------------------------------------------------

def plot_method_comparison(list_total_policies,path,comp_type='scores',reward_threshold=None):
    colors=list(mcolors.TABLEAU_COLORS.keys())
    plt.figure(figsize=[8,5])
    plt.subplots_adjust(left=0.07,bottom=0.152,right=0.95,top=0.94,wspace=0.39,hspace=0.2)
    n_plot=1
    for i in list_total_policies:

        ax=plt.subplot(2,int(len(list_total_policies)/2),n_plot)
        ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

        #SH
        x,y_scores,y_policies=tradeoff_curve_successive_halving(i,2)
        if comp_type=='scores':
            y=y_scores
        else:
            y=y_policies
        ax.fill_between(x,y[1],y[2], alpha=.2, linewidth=0,color=colors[n_plot-1])
        plt.plot(x, y[0], linewidth=1,marker='.',color=colors[n_plot-1],label='SH')

        #PSH soft 
        x,y_scores,y_policies=tradeoff_curve_successive_halving(i,2,'PSH')
        if comp_type=='scores':
            y=y_scores
        else:
            y=y_policies
        ax.fill_between(x,y[1],y[2], alpha=.2, linewidth=0,color=colors[n_plot-1])
        plt.plot(x, y[0], linewidth=1,marker='^',color=colors[n_plot-1],label='PSH soft')
        #PSH original
        x,y_scores,y_policies=tradeoff_curve_successive_halving(i,2,'PSH',[False,None])
        if comp_type=='scores':
            y=y_scores
        else:
            y=y_policies
        ax.fill_between(x,y[1],y[2], alpha=.2, linewidth=0,color=colors[n_plot-1])
        plt.plot(x, y[0], linewidth=1,marker='x',color=colors[n_plot-1],label='PSH original')
        #RS
        x_prev=x
        x,y_scores,y_policies=curve_tradeoff_random_search(i,max(x_prev))
        if comp_type=='scores':
            y=y_scores
        else:
            y=y_policies
        y_mean=[y[0][j-1]for j in x_prev]
        y_q05=[y[1][j-1]for j in x_prev]
        y_q95=[y[2][j-1]for j in x_prev]
        ax.fill_between(x_prev,y_q05,y_q95, alpha=.2, linewidth=0,color=colors[n_plot-1])
        plt.plot(x_prev, y_mean, linewidth=1,marker='D',color=colors[n_plot-1],label='RS')

        if n_plot in [1,4]:
            if comp_type=='scores':
                ax.set_ylabel("Test reward",fontsize=8)
            else:
                ax.set_ylabel("n_policies used",fontsize=8)
        if n_plot==2:
            ax.set_title('Trade-off curve comparison')
        if n_plot==5:
            ax.legend(title="Method",fontsize=8,ncol=6,bbox_to_anchor=(1.5, -0.1, 0, 0))

        if reward_threshold is not None:
            plt.axhline(y=reward_threshold,color='black', linestyle='--')

        n_plot+=1

    plt.savefig(path)
    plt.show()
    plt.close()



def plot_tradeoff_curves(list_total_policies,plot_type,title,path,reward_threshold,extra_var=[None,None,None]):
    plt.figure(figsize=[12,4])
    plt.subplots_adjust(left=0.1,bottom=0.11,right=0.95,top=0.88,wspace=0.97,hspace=0.2)

    # Trade-off curves per n_policy
    ax=plt.subplot(2,6,(3,10))
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    y_best_n_process=[]
    y_best_score=[]
    for i in list_total_policies:

        if plot_type=='random_search':
            x,y,_=curve_tradeoff_random_search(i,extra_var)

        if plot_type=='successive_halving':
            if extra_var[0] is not None:
                x,y,_=tradeoff_curve_successive_halving(i,2,extra_var[0],extra_var[1:])
            else:
                x,y,_=tradeoff_curve_successive_halving(i,2)
        
        index_max=[i for i in range(len(y[0])) if y[0][i]==max(y[0])]
        y_best_n_process.append(max(np.array(x)[index_max]))
        y_best_score.append(max(y[0]))

        ax.fill_between(x,y[1],y[2], alpha=.5, linewidth=0)
        plt.plot(x, y[0], linewidth=1,label=str(i),marker='.')


    plt.axhline(y=reward_threshold,color='black', linestyle='--')
    ax.set_xlabel("n_processes",fontsize=10)
    ax.set_ylabel("Best test reward",fontsize=10)
    ax.set_title(title)
    ax.legend(title="n_policies",fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Trade-off n_processes per n_policy
    str_x=[str(i) for i in list_total_policies]
    str_x.reverse()
    y_best_n_process.reverse()
    y_best_score.reverse()
    colors=list(mcolors.TABLEAU_COLORS.keys())[:len(str_x)]
    colors.reverse()

    ax=plt.subplot(2,6,(5,6))
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)
    ax.bar(str_x, y_best_n_process,width=0.7,color=colors)
    ax.set_xlabel("")
    ax.set_ylabel("Best n_processes")
    ax.set_title('')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Trade-off best reward per n_policy
    ax=plt.subplot(2,6,(11,12))
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)
    ax.bar(str_x,y_best_score ,width=0.7,color=colors)
    ax.set_xlabel("n_policies")
    ax.set_ylabel("Best test reward")
    ax.set_title('')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Learning-curves
    ax=plt.subplot(2,6,(1,8))
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    x=list(range(1,total_policies+1))
    y_matrix=[]

    for seed in range(1,41):
        y_matrix.append(list(df[(df['seed']==seed) & (df['n_policy']<=total_policies)]['mean_reward']))

    y=from_matrix_to_ycoordinates(np.array(y_matrix).T)

    ax.fill_between(x,y[1],y[2], alpha=.5, linewidth=0,color='black')
    plt.plot(x, y[0], linewidth=1,color='black')
    ax.set_xlabel("n_policies")
    ax.set_ylabel("Test reward")
    ax.set_title('Mean learning-curve')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.savefig(path)
    plt.show()
    plt.close()

#--------------------------------------------------------------------------------------------------
# Random search
#--------------------------------------------------------------------------------------------------
# Para dibujar una curva trade-off con n_policy fijo
def curve_tradeoff_random_search(total_policies,max_n_processes):

    list_n_policy=[]
    for i in range(max_n_processes):
        list_n_policy.append(total_policies//(i+1))

    scores=[]
    policies=[]
    for i in range(max_n_processes):
        list_scores=[]
        list_policies=[]
        list_seeds_sub=n_subsets_of_fixed_len(exp_seeds,n_process,i+1)

        for seeds_sub in list_seeds_sub:
            # Successive halving
            df_current=df[(df['n_policy']==list_n_policy[i]) & (df['seed'].isin(seeds_sub))]
            list_scores.append(max(list(df_current['mean_reward'])))
            list_policies.append(list_n_policy[i]*len(seeds_sub))
        
        scores.append(list_scores)
        policies.append(list_policies)

    return list(range(1,max_n_processes+1)),from_matrix_to_ycoordinates(np.array(scores)),from_matrix_to_ycoordinates(np.array(policies))

#--------------------------------------------------------------------------------------------------
# Successive halving
#--------------------------------------------------------------------------------------------------
# Original
def successive_halving(list_processes,r,eta,s):
    policies=0

    for i in range(s+1):
        r_i=math.floor(r*(eta**i))

        policies+=r_i*len(list_processes)

        df_current=df[(df['n_policy']==r_i) & (df['seed'].isin(list_processes))]
        list_scores=list(df_current['mean_reward'])
        list_scores.sort(reverse=True)

        if i<s:
            n_next=math.floor(len(list_processes)/eta)
            best_scores=list_scores[:n_next]
            list_processes=list(df[(df['n_policy']==r_i) & (df['seed'].isin(list_processes)) & (df['mean_reward'].isin(best_scores))]['seed'])[:n_next]
        else:
            best_score=list_scores[0]
 
    return best_score,policies

#Progressive successive halving
def progressive_successive_halving(list_processes,r,eta,s,soft_stability=True,epsilon='2std'):
    stable=False
    policies=0

    for i in range(s+1):
        r_i=math.floor(r*(eta**i))
        policies+=r_i*len(list_processes)

        df_current=df[(df['n_policy']==r_i) & (df['seed'].isin(list_processes))]
        list_scores=list(df_current['mean_reward'])     
        list_seeds=list(df_current['seed'])

        
        if i<s:
            n_next=math.floor(len(list_processes)/eta)

            # Decidir si seguir dividiendo
            if i==0:

                previous_ranking=sort_list_according_to_argsort_list(list_seeds,np.argsort(-np.array(list_scores)))

            else:
                current_ranking=sort_list_according_to_argsort_list(list_seeds,np.argsort(-np.array(list_scores)))
             
                if not soft_stability:
                    # Comprobar estabilidad de ranking originales
                    if previous_ranking[:len(current_ranking)]==current_ranking:
                        stable=True
                else:

                    # Soft ranking
                    list_scores.sort(reverse=True)
                    soft_ranking={key:[key] for key in range(len(current_ranking))}
                    posible_pairs=list(itertools.permutations(list(range(len(current_ranking))), 2))
                    dist_list=[abs(list_scores[pair[0]]-list_scores[pair[1]]) for pair in posible_pairs]
                    epsilon=np.std(dist_list)*2
                    for pair in posible_pairs:
                        if dist_list[posible_pairs.index(pair)]<=epsilon:
                            soft_ranking[pair[0]].append(current_ranking[pair[1]])
                            soft_ranking[pair[1]].append(current_ranking[pair[0]])

                    # Comprobar estabilidad soft rankings
                    not_soft=False
                    for i in range(len(current_ranking)):
                        if previous_ranking[i] not in soft_ranking[i]:
                            not_soft=True
                    if not_soft is not True:
                        stable=True

                previous_ranking=current_ranking

            if stable:
                best_score=list_scores[0]
                break
            else:
                # Escoger los mejores
                list_scores.sort(reverse=True)
                best_scores=list_scores[:n_next]

                list_processes=list(df[(df['n_policy']==r_i) & (df['seed'].isin(list_processes)) & (df['mean_reward'].isin(best_scores))]['seed'])[:n_next]
        else:
            best_score=list_scores[0]

                
    return best_score,policies

# Para dibujar una curva trade-off con n_policy fijo
def tradeoff_curve_successive_halving(total_policies,eta,type='SH',extra_var=None):

    '''
    Inplementation based on:
    -https://arxiv.org/abs/1603.06560
    -https://arxiv.org/abs/2207.06940
    '''

    def compute_s(B, eta):
        input_s = 0 
        output_s=None
        while output_s!=input_s:
            output_s = math.floor(math.log(B / (input_s + 1), eta))
            input_s+=1
            if output_s<0:
                raise ValueError( 'SH cannot be applied with total_policies='+str(total_policies)+' and eta='+str(eta)+' value combination')

        return input_s


    s_max=compute_s(total_policies,eta)
    R=total_policies/(s_max+1)

    list_n=[]
    scores=[]
    policies=[]

    for s in range(s_max+1):

        n=math.ceil((total_policies/R)*((eta**s)/(s+1)))
        r=R/(eta**s)

        if r>1:
            list_n.append(n)

            list_scores=[]
            list_policies=[]
            list_seeds_sub=n_subsets_of_fixed_len(exp_seeds,n_process,n)
            for seed_sub in list_seeds_sub:
                if type=='SH':
                    current_score,current_policies=successive_halving(seed_sub,r,eta,s)
                if type=='PSH':
                    if extra_var is None:
                        current_score,current_policies=progressive_successive_halving(seed_sub,r,eta,s)
                    else:
                        current_score,current_policies=progressive_successive_halving(seed_sub,r,eta,s,extra_var[0],extra_var[1])

                list_scores.append(current_score)
                list_policies.append(current_policies)

            scores.append(list_scores)
            policies.append(list_policies)

    return list_n,from_matrix_to_ycoordinates(np.array(scores)),from_matrix_to_ycoordinates(np.array(policies))

#==================================================================================================
# Programa principal
#==================================================================================================

random.seed(0)
np.random.seed(0)


n_process=40
t_r=2048
data_path='experiments_getstarting/results/ResourceAllocators/'
exp_seeds=list(range(1,41))

# Dibujar graficas para "InvertedDoublePendulum"
t_max=500000
total_policies=t_max//t_r
list_total_policies=[240,170,127,95,45,24]
env_name='InvertedDoublePendulum'
df=concat_df(list(range(1,41)),env_name, data_path+env_name+'/')

plot_tradeoff_curves(list_total_policies,'random_search','Trade-off curves with RS','experiments_getstarting/results/ResourceAllocators/'+str(env_name)+'RS.pdf',9350,10)
plot_tradeoff_curves(list_total_policies,'successive_halving','Trade-off curves with SH','experiments_getstarting/results/ResourceAllocators/'+str(env_name)+'SH.pdf',9350)
plot_tradeoff_curves(list_total_policies,'successive_halving','Trade-off curves with PSH\n(soft ranking stability)','experiments_getstarting/results/ResourceAllocators/'+str(env_name)+'PSH_soft.pdf',9350,['PSH',None,None])
plot_tradeoff_curves(list_total_policies,'successive_halving','Trade-off curves with PSH\n(original ranking stability)','experiments_getstarting/results/ResourceAllocators/'+str(env_name)+'PSH_original.pdf',9350,['PSH',False,None])
plot_method_comparison(list_total_policies,'experiments_getstarting/results/ResourceAllocators/'+env_name+'Comparison_scores.pdf')
plot_method_comparison(list_total_policies,'experiments_getstarting/results/ResourceAllocators/'+env_name+'Comparison_time.pdf',comp_type='time')

# Dibujar graficas para "InvertedPendulum"
t_max=200000
total_policies=t_max//t_r
list_total_policies=[95,45,30,24,15,10]
env_name='InvertedPendulum'
df=concat_df(list(range(1,41)),env_name, data_path+env_name+'/')

plot_tradeoff_curves(list_total_policies,'random_search','Trade-off curves with RS','experiments_getstarting/results/ResourceAllocators/'+str(env_name)+'RS.pdf',1000,10)
plot_tradeoff_curves(list_total_policies,'successive_halving','Trade-off curves with SH','experiments_getstarting/results/ResourceAllocators/'+str(env_name)+'SH.pdf',1000)
plot_tradeoff_curves(list_total_policies,'successive_halving','Trade-off curves with PSH\n(soft ranking stability)','experiments_getstarting/results/ResourceAllocators/'+str(env_name)+'PSH_soft.pdf',1000,['PSH',None,None])
plot_tradeoff_curves(list_total_policies,'successive_halving','Trade-off curves with PSH\n(original ranking stability)','experiments_getstarting/results/ResourceAllocators/'+str(env_name)+'PSH_original.pdf',1000,['PSH',False,None])
plot_method_comparison(list_total_policies,'experiments_getstarting/results/ResourceAllocators/'+env_name+'Comparison_scores.pdf')
plot_method_comparison(list_total_policies,'experiments_getstarting/results/ResourceAllocators/'+env_name+'Comparison_time.pdf',comp_type='time')
