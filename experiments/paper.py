import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Regiones

def learning_region_limits(truth_last):

    a,b=None,None
    for i in range(len(truth_last)):

        if truth_last[i]-min(truth_last[:i+1])<=np.std(truth_last[:i+1]) :
            if a==None or i-a<int(len(truth_last)*0.1): # La condicion tiene que cumplirse con una cierta continuidad (no vale que se vuelva a cumprir mucho despues otra vez)
                a=i
        if max(truth_last[-i-1:])-truth_last[-1-i]<=np.std(truth_last[-i-1:]) :
            if b==None or b-len(truth_last)+i<int(len(truth_last)*0.1):
                b=len(truth_last)-i


    # Para saber los limites que cumples
    a_std,b_std=a,b
    if a<int(0.1*len(truth_last)):
        print('a para garantizar minimo size de [0,a]')
    if b>len(truth_last)-int(0.1*len(truth_last)):
        print('b para garantizar minimo size de [b,T]')

    if max([a,int(0.1*len(truth_last))])>int(2*len(truth_last)/3)-int(0.1*len(truth_last)):
        print('a para no pasarnos de t_max')
    if min([b,len(truth_last)-int(0.1*len(truth_last))])<a+int(0.1*len(truth_last)):
        print('b para garantizar a>b')

    # Para que siempre haya un minimo de datos en los intervalos [0,a], [a,b] y [b,T]
    a=min([max([a,int(0.1*len(truth_last))]),int(2*len(truth_last)/3)-int(0.1*len(truth_last))])
    b=max([min([b,len(truth_last)-int(0.1*len(truth_last))]),a+int(0.1*len(truth_last))])

    # Para saber los limites que cumples
    if a==a_std:
        print('a con std')
    if b==b_std:
        print('b con std')

    return a,b,int(2*len(truth_last)/3)

def plot_curve_with_limits(ax,truth_last,first=False):

    a,b,t_max=learning_region_limits(truth_last)


    ax.plot(range(len(truth_last)), truth_last,color='blue', linewidth=1)  
    ax.axvline(a, color='red',linewidth=2,label=r'$a$')            
    ax.axvline(b, color='black',linewidth=2,label=r'$b$')        
    ax.axvline(t_max, linestyle='--',color='black', linewidth=1,label=r'$t_{max}$')
    if first:
        ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=3,
    frameon=False
)
    
fig, axs = plt.subplots(1,4, figsize=(12, 2.5))
plt.subplots_adjust(top=0.9,bottom=0.3,left=0.05,right=0.98, hspace=0.0,wspace=0.2)

truth_last1=pd.read_csv('experiments/results/data/paper/a_b_std.csv').iloc[:, 0].tolist()
truth_last2=pd.read_csv('experiments/results/data/paper/a_b_min.csv').iloc[:, 0].tolist()
truth_last3=pd.read_csv('experiments/results/data/paper/a_min.csv').iloc[:, 0].tolist()
truth_last4=pd.read_csv('experiments/results/data/paper/b_max.csv').iloc[:, 0].tolist()

plot_curve_with_limits(axs[0],truth_last1,first=True)
plot_curve_with_limits(axs[1],truth_last2)
plot_curve_with_limits(axs[2],truth_last3)
plot_curve_with_limits(axs[3],truth_last4)

axs[0].set_ylabel(r'$f(\pi_t)$',fontsize=10)
for i in range(4):
    axs[i].set_xlabel(r'$t$',fontsize=10)

plt.savefig('experiments/results/figures/paper/regions.pdf')


fig, axs = plt.subplots(1,3, figsize=(12, 2.5))
plt.subplots_adjust(top=0.9,bottom=0.2,left=0.05,right=0.98, hspace=0.0,wspace=0.2)

# Degradacion
truth_last=pd.read_csv('experiments/results/data/paper/a_b_std.csv').iloc[:, 0].tolist()

def degradation_evolution(truth_last):
    deg_evol=[]
    for i in range(len(truth_last)):
        best_truth=max(truth_last[:i+1])
        last_truth=truth_last[i]
        current_mean_truth=np.mean(truth_last[:i+1])

        if best_truth!=last_truth:
            degradation_level=(best_truth-last_truth)/(best_truth-current_mean_truth)
        else:
            degradation_level=0

        deg_evol.append(degradation_level)
    return deg_evol

def plot_degradation(ax,truth_last):
    deg_evol=degradation_evolution(truth_last)

    ax.plot(range(len(truth_last)),truth_last,color='blue',linewidth=1)

    #t=39, no hay degradacion
    ax.axhline(y=np.mean(truth_last[:39+1]), color='purple', linestyle='-',alpha=0.5)
    ax.scatter(39, truth_last[39], color='purple')
    ax.text(0.02, 0.15,r'$\text{mean}_{0,39}$',transform=ax.get_yaxis_transform(),color='purple',va='bottom')
    ax.text(39-20, truth_last[39]+0.1,r'$\delta_{39}=$'+str(deg_evol[39]), va='center',color='purple')

    #t=70, hay degradacion
    ax.axhline(y=np.mean(truth_last[:66+1]), color='red', linestyle='-',alpha=0.5)
    ax.scatter(66, truth_last[66], color='red')
    ax.text(0.02, 0.35,r'$\text{mean}_{0,66}$',transform=ax.get_yaxis_transform(),color='red',va='bottom')
    ax.text(66, truth_last[66]-0.1,r'$\delta_{66}=$'+str(round(deg_evol[66],2)), va='center',color='red')

    ax.set_title('Degradation',fontsize=10)
    ax.set_ylabel(r'$f(\pi_t)$',fontsize=10)
    ax.set_xlabel(r'$t$',fontsize=10)

plot_degradation(axs[0],truth_last)

# Precision
def precision(truth_selection,truth_best,min_truth):
    return (truth_selection-min_truth)/(truth_best-min_truth)

def plot_precision(ax,truth_last):

    ax.plot(range(65+1),truth_last[:65+1],color='blue',linewidth=1)
    ax.plot(range(65,100),truth_last[65:101],color='blue',linestyle='--',linewidth=1,alpha=0.3)
    ax.axhline(y=min(truth_last[:65+1]), color='red', linestyle='-',alpha=0.5)
    ax.text(0.7, 0.01,r'$\text{min}_{0,t}$',transform=ax.get_yaxis_transform(),color='red',va='bottom')

    #t=56, politica seleccionada por criterio
    ax.scatter(44, truth_last[44], color='purple')
    ax.text(44-5, truth_last[44]+0.1,r'$\widetilde{\pi}_{t}$', va='center',color='purple')
    ax.vlines(x=44, ymin=0, ymax=truth_last[44], color='purple',linestyle='--',linewidth=1)

    ax.scatter(truth_last.index(max(truth_last[:65+1])), max(truth_last[:65+1]), color='black')
    ax.text(truth_last.index(max(truth_last[:65+1])), max(truth_last[:65+1])+0.1,r'$\pi^*_{t}$', va='center',color='black')
    ax.vlines(x=truth_last.index(max(truth_last[:65+1])), ymin=0, ymax=max(truth_last[:65+1]), color='black',linestyle='--',linewidth=1)

    ax.scatter(truth_last.index(max(truth_last)), max(truth_last), color='grey')
    ax.text(truth_last.index(max(truth_last))+2, max(truth_last)-0.35,r"$\pi^*_{t+t'}$", va='center',color='grey')
    ax.vlines(x=truth_last.index(max(truth_last)), ymin=0, ymax=max(truth_last), color='grey',linestyle='--',linewidth=1)

    prec1=round(precision(truth_last[44],max(truth_last[:65+1]),min(truth_last[:65+1])),2)
    prec2=round(precision(truth_last[44],max(truth_last),min(truth_last)),2)
    ax.text(0.02, 0.98,f'Train: $\\alpha_t$ = {prec1}\n'
    f'Test: $\\alpha_t$ = {prec2}\n'
    r'$t=65,\ t^\prime=35$',
    transform=ax.transAxes,ha='left',va='top',fontsize=8)

    ax.set_title('Accuracy',fontsize=10)
    ax.set_xlabel(r'$t$',fontsize=10)

plot_precision(axs[1],truth_last)

# Eficiencia
def efficiency(truth_selection,all_truth):
    indice = next(i for i, x in enumerate(all_truth) if x >= truth_selection)
    return indice,(indice+1)/len(all_truth)

def plot_efficiency(ax,truth_last):

    ax.plot(range(len(truth_last)),truth_last,color='blue',linewidth=1)

    #t=61, eficiencia baja por criterio
    t_min,eff=efficiency(truth_last[61],truth_last)
    ax.scatter(61, truth_last[61], color='orange')
    ax.text(61, truth_last[61]-0.25,r'$\widetilde{\pi}_{t}$',color='orange',va='bottom')
    ax.scatter(t_min, truth_last[61], color='orange',alpha=0.5)
    ax.hlines(y=truth_last[61], xmin=0, xmax=t_min, color='orange',linestyle='--')
    ax.text(0.05, truth_last[61]+0.01,r'$\varepsilon_{t}=$'+str(round(eff,2)),transform=ax.get_yaxis_transform(),color='orange',va='bottom')


    #t=92, eficiencia baja por convergencia
    t_min,eff=efficiency(truth_last[92],truth_last)
    ax.scatter(92, truth_last[92], color='purple')
    ax.text(92, truth_last[92]-0.3,r'$\widetilde{\pi}_{t}$',color='purple',va='bottom')
    ax.scatter(t_min, truth_last[92], color='purple',alpha=0.5)
    ax.hlines(y=truth_last[92], xmin=0, xmax=t_min, color='purple',linestyle='--')
    ax.text(0.05, truth_last[92]-0.15,r'$\varepsilon_{t}=$'+str(round(eff,2)),transform=ax.get_yaxis_transform(),color='purple',va='bottom')

    ax.set_title('Efficiency',fontsize=10)
    ax.set_xlabel(r'$t$',fontsize=10)

plot_efficiency(axs[2],truth_last)
plt.savefig('experiments/results/figures/paper/deg_acc_eff.pdf')