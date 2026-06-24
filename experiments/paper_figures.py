import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.lines as mlines

#==================================================================================================
# Regiones
#==================================================================================================

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


#==================================================================================================
# Degradacion, precision y eficacia 
#==================================================================================================

fig, axs = plt.subplots(1,3, figsize=(12, 3))
plt.subplots_adjust(top=0.9,bottom=0.3,left=0.05,right=0.98, hspace=0.0,wspace=0.2)

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

    def draw_deg(iter):
        ax.axhline(y=np.mean(truth_last[:iter+1]), color='red',linewidth=1, linestyle='-',alpha=0.5)
        ax.scatter(iter, truth_last[iter], color='black')
        ax.scatter(truth_last[:iter+1].index(max(truth_last[:iter+1])), max(truth_last[:iter+1]), color='black')
        ax.text(0.02, np.mean(truth_last[:iter+1]),rf'$\text{{mean}}_{{0,{iter}}}$',transform=ax.get_yaxis_transform(),color='red',va='bottom')
        ax.text(iter, truth_last[iter]-0.1, rf'$\pi_{{{iter}}}$', va='center',color='black')
        ax.text(truth_last[:iter+1].index(max(truth_last[:iter+1])), max(truth_last[:iter+1])+0.1, rf'$\pi^*_{{{iter}}}$', va='center',color='black')

        ax.vlines(x=truth_last[:iter+1].index(max(truth_last[:iter+1])), ymin=max(truth_last[:iter+1]), ymax=np.mean(truth_last[:iter+1]), color='grey',linestyle='-',linewidth=5,alpha=0.3)
        ax.vlines(x=truth_last[:iter+1].index(max(truth_last[:iter+1])), ymin=max(truth_last[:iter+1]), ymax=truth_last[iter], color="black",linestyle='-',linewidth=5,alpha=0.5)

        ax.plot([truth_last[:iter+1].index(max(truth_last[:iter+1])),iter],
                [truth_last[iter] ]*2, color='black',linewidth=1,linestyle='--')

    ax.plot(range(len(truth_last)),truth_last,color='blue',linewidth=1)

    #t=39, no hay degradacion
    ax.axhline(y=np.mean(truth_last[:39+1]), color='red',linewidth=1, linestyle='-',alpha=0.5)
    ax.scatter(39, truth_last[39], color='black')
    ax.text(0.02, 0.15,r'$\text{mean}_{0,39}$',transform=ax.get_yaxis_transform(),color='red',va='bottom')
    ax.text(39-20, truth_last[39]+0.1,r'$\pi^*_{39}=\pi_{39}$', va='center',color='black')

    ax.vlines(x=39, ymin=np.mean(truth_last[:39+1]), ymax=max(truth_last[:39+1]), color='grey',linestyle='-',linewidth=5,alpha=0.3)


    #t=66, hay degradacion
    draw_deg(66)
    
    #t=89, hay degradacion
    draw_deg(89)

    text = (
    rf'$\delta_{{39}} = {deg_evol[39]}$' + '\n' +
    rf'$\delta_{{66}} = {round(deg_evol[66],2)}$' + '\n' +
    rf'$\delta_{{89}} = {round(deg_evol[89],2)}$'
)

    ax.text(0.02, 0.98,text,
    transform=ax.transAxes,ha='left',va='top',fontsize=10)


    ax.set_title('Degradation',fontsize=10)
    ax.set_ylabel(r'$f(\pi_t)$',fontsize=10)
    ax.set_xlabel(r'$t$',fontsize=10)
    ax.set_ylim([-0.1, 1.3])
    ax.set_yticks(np.arange(0, 1.01, 0.25))

plot_degradation(axs[0],truth_last)

# Precision
def precision(truth_selection,truth_best,min_truth):
    return (truth_selection-min_truth)/(truth_best-min_truth)

def plot_precision(ax,truth_last):

    ax.plot(range(65+1),truth_last[:65+1],color='blue',linewidth=1)
    ax.plot(range(65,100),truth_last[65:101],color='blue',linestyle='--',linewidth=1,alpha=0.3)
    ax.axhline(y=min(truth_last[:65+1]), color='red', linestyle='-',alpha=0.5)
    ax.text(0.65, 0.01,r'$\text{min}_{0,65}$',transform=ax.get_yaxis_transform(),color='red',va='bottom')

    #t=35, politica seleccionada por criterio
    ax.scatter(35, truth_last[35], color='orange')
    ax.text(35-8, truth_last[35]+0.1,r'$\widetilde{\pi}_{65}$', va='center',color='orange')
    ax.hlines(y=truth_last[35], xmin=35, xmax=truth_last[:65+1].index(max(truth_last[:65+1])), color='orange',linestyle='--',linewidth=1)


    ax.scatter(50, truth_last[50], color='green')
    ax.text(50-8, truth_last[50]+0.1,r'$\widetilde{\pi}_{65}$', va='center',color='green')
    ax.hlines(y=truth_last[50], xmin=50, xmax=truth_last.index(max(truth_last)), color='green',linestyle='--',linewidth=1)


    ax.scatter(truth_last.index(max(truth_last[:65+1])), max(truth_last[:65+1]), color='black')
    ax.text(truth_last.index(max(truth_last[:65+1])), max(truth_last[:65+1])+0.1,r'$\pi^*_{65}$', va='center',color='black')
    ax.vlines(x=truth_last.index(max(truth_last[:65+1])), ymin=min(truth_last[:65+1]), ymax=max(truth_last[:65+1]), color='grey',linestyle='-',linewidth=5,alpha=0.3)
    ax.vlines(x=truth_last.index(max(truth_last[:65+1])), ymin=min(truth_last[:65+1]), ymax=truth_last[35], color="orange",linestyle='-',linewidth=5,alpha=0.5)


    ax.scatter(truth_last.index(max(truth_last)), max(truth_last), color='black')
    ax.text(truth_last.index(max(truth_last)), max(truth_last)+0.1,r"$\pi^*_{100}$", va='center',color='black')
    ax.vlines(x=truth_last.index(max(truth_last)), ymin=min(truth_last[:65+1]), ymax=max(truth_last), color='grey',linestyle='-',linewidth=5,alpha=0.3)
    ax.vlines(x=truth_last.index(max(truth_last)), ymin=min(truth_last[:65+1]), ymax=truth_last[50], color="green",linestyle='-',linewidth=5,alpha=0.5)


    prec1=round(precision(truth_last[35],max(truth_last[:65+1]),min(truth_last[:65+1])),2)
    prec2=round(precision(truth_last[50],max(truth_last),min(truth_last)),2)
    ax.text(0.02, 0.98,rf'Train: $\alpha_{{65}}$ = {prec1}',transform=ax.transAxes,ha='left',va='top',fontsize=10,color='orange')
    ax.text(0.02, 0.89,rf'Test: $\alpha_{{65}}$ = {prec2}',transform=ax.transAxes,ha='left',va='top',fontsize=10,color='green')
    ax.text(0.02, 0.80,r'$(t^\prime = 35)$',transform=ax.transAxes,ha='left',va='top',fontsize=10,color='green')

    ax.set_title('Accuracy',fontsize=10)
    ax.set_xlabel(r'$t$',fontsize=10)

    ax.set_ylim([-0.1, 1.3])
    ax.set_yticks(np.arange(0, 1.01, 0.25))

plot_precision(axs[1],truth_last)

# Eficiencia
def efficiency(truth_selection,all_truth):
    indice = next(i for i, x in enumerate(all_truth) if x >= truth_selection)
    return indice,(indice+1)/len(all_truth)

def plot_efficiency(ax,truth_last):

    ax.plot(range(len(truth_last)),truth_last,color='blue',linewidth=1)

    ax.axvline(x=100, color='red', linestyle='-',alpha=0.5)

    #t=61, eficiencia baja por criterio
    t_min,eff=efficiency(truth_last[61],truth_last)
    ax.hlines(y=truth_last[61], xmin=0, xmax=100, color="grey",linestyle='-',linewidth=5,alpha=0.3,zorder=1)
    ax.hlines(y=truth_last[61], xmin=0, xmax=t_min, color='orange',linestyle='-',linewidth=5,alpha=0.5,zorder=2)
    ax.scatter(61, truth_last[61], color='orange',zorder=10)
    ax.text(61-2, truth_last[61]-0.2,r'$\widetilde{\pi}_{100}$',color='orange',va='bottom')
    ax.scatter(t_min, truth_last[61], color='black',marker='x',zorder=10)
    ax.text(0.05, truth_last[61]+0.03,r'$\varepsilon_{100}=$'+str(round(eff,2)),transform=ax.get_yaxis_transform(),color='orange',va='bottom')


    #t=92, eficiencia baja por convergencia
    t_min,eff=efficiency(truth_last[92],truth_last)
    ax.hlines(y=truth_last[92], xmin=0, xmax=100, color="grey",linestyle='-',linewidth=5,alpha=0.3,zorder=1)
    ax.hlines(y=truth_last[92], xmin=0, xmax=t_min, color='green',linestyle='-',linewidth=5,alpha=0.5,zorder=2)
    ax.scatter(92, truth_last[92], color='green',zorder=10)
    ax.text(92-2, truth_last[92]-0.3,r'$\widetilde{\pi}_{100}$',color='green',va='bottom')
    ax.scatter(t_min, truth_last[92], color='black',zorder=10,marker='x')
    ax.text(0.05, truth_last[92]-0.15,r'$\varepsilon_{100}=$'+str(round(eff,2)),transform=ax.get_yaxis_transform(),color='green',va='bottom')

    ax.set_title('Efficiency',fontsize=10)
    ax.set_xlabel(r'$t$',fontsize=10)

    # Leyenda
    legend_handles = [
    mlines.Line2D([], [], color='grey', linestyle='-', linewidth=6, alpha=0.3,label='denominator of metric'),
    mlines.Line2D([], [], color='black', linestyle='-', linewidth=6, alpha=0.5,label=r'numerator of $\delta_t$'),
    mlines.Line2D([], [], color='orange', linestyle='-', linewidth=6, alpha=0.5,label=r'numerator of $\alpha_t$ or $\varepsilon_t$ for selection 1'),
    mlines.Line2D([], [], color='green', linestyle='-', linewidth=6, alpha=0.5,label=r'numerator of $\alpha_t$ or $\varepsilon_t$ for selection 2'),
    mlines.Line2D([], [], color='red', linestyle='-', linewidth=2,label='reference value for metric'),
    # mlines.Line2D([], [], color='orange', marker='o', linestyle='None', markersize=7, label='selection 1'),
    # mlines.Line2D([], [], color='green', marker='o', linestyle='None',markersize=7, label='selection 2'),
    mlines.Line2D([], [], color='blue', linestyle='--',linewidth=2, label=r"performance evolution by investing validation time $t'$ also learning"),
    mlines.Line2D([], [], color='black', marker='x', linestyle='None',markersize=7, label='first time with same performance'),]

    leg1 = ax.legend(handles=legend_handles[:4],loc='upper center',bbox_to_anchor=(-0.75, -0.2),ncol=6,frameon=False)
    leg2 = ax.legend(handles=legend_handles[4:],loc='upper center',bbox_to_anchor=(-0.75, -0.32),ncol=6,frameon=False)
    ax.add_artist(leg1)

plot_efficiency(axs[2],truth_last)
plt.savefig('experiments/results/figures/paper/deg_acc_eff.pdf')



#==================================================================================================
# Degradacion, precision y eficacia (MINDIVIDUALES)
#==================================================================================================



# Degradacion
fig, axs = plt.subplots(1,1, figsize=(4.5, 3))
plt.subplots_adjust(top=0.93,bottom=0.25,left=0.15,right=0.98, hspace=0.0,wspace=0.2)

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

    def draw_deg(iter):
        ax.axhline(y=np.mean(truth_last[:iter+1]), color='red',linewidth=1, linestyle='-',alpha=0.5)
        ax.scatter(iter, truth_last[iter], color='black')
        ax.scatter(truth_last[:iter+1].index(max(truth_last[:iter+1])), max(truth_last[:iter+1]), color='black')
        ax.text(0.02, np.mean(truth_last[:iter+1]),rf'$\text{{mean}}_{{0,{iter}}}$',transform=ax.get_yaxis_transform(),color='red',va='bottom')
        ax.text(iter, truth_last[iter]-0.1, rf'$\pi_{{{iter}}}$', va='center',color='black')
        ax.text(truth_last[:iter+1].index(max(truth_last[:iter+1])), max(truth_last[:iter+1])+0.1, rf'$\pi^*_{{{iter}}}$', va='center',color='black')

        ax.vlines(x=truth_last[:iter+1].index(max(truth_last[:iter+1])), ymin=max(truth_last[:iter+1]), ymax=np.mean(truth_last[:iter+1]), color='grey',linestyle='-',linewidth=5,alpha=0.3)
        ax.vlines(x=truth_last[:iter+1].index(max(truth_last[:iter+1])), ymin=max(truth_last[:iter+1]), ymax=truth_last[iter], color="black",linestyle='-',linewidth=5,alpha=0.5)

        ax.plot([truth_last[:iter+1].index(max(truth_last[:iter+1])),iter],
                [truth_last[iter] ]*2, color='black',linewidth=1,linestyle='--')

    ax.plot(range(len(truth_last)),truth_last,color='blue',linewidth=1)

    #t=39, no hay degradacion
    ax.axhline(y=np.mean(truth_last[:39+1]), color='red',linewidth=1, linestyle='-',alpha=0.5)
    ax.scatter(39, truth_last[39], color='black')
    ax.text(0.02, 0.15,r'$\text{mean}_{0,39}$',transform=ax.get_yaxis_transform(),color='red',va='bottom')
    ax.text(39-20, truth_last[39]+0.1,r'$\pi^*_{39}=\pi_{39}$', va='center',color='black')

    ax.vlines(x=39, ymin=np.mean(truth_last[:39+1]), ymax=max(truth_last[:39+1]), color='grey',linestyle='-',linewidth=5,alpha=0.3)


    #t=66, hay degradacion
    draw_deg(66)
    
    #t=89, hay degradacion
    draw_deg(89)

    text = (
    rf'$\delta_{{39}} = {deg_evol[39]}$' + '\n' +
    rf'$\delta_{{66}} = {round(deg_evol[66],2)}$' + '\n' +
    rf'$\delta_{{89}} = {round(deg_evol[89],2)}$'
)

    ax.text(0.02, 0.98,text,
    transform=ax.transAxes,ha='left',va='top',fontsize=10)


    ax.set_title('Degradation',fontsize=10)
    ax.set_ylabel(r'$f(\pi_t)$',fontsize=10)
    ax.set_xlabel(r'$t$',fontsize=10)
    ax.set_ylim([-0.1, 1.3])
    ax.set_yticks(np.arange(0, 1.01, 0.25))

    # Leyenda
    legend_handles = [
    mlines.Line2D([], [], color='grey', linestyle='-', linewidth=6, alpha=0.3,label=r'denominator of $\delta_t$'),
    mlines.Line2D([], [], color='black', linestyle='-', linewidth=6, alpha=0.5,label=r'numerator of $\delta_t$'),
]
    leg1 = ax.legend(handles=legend_handles[:4],loc='upper center',bbox_to_anchor=(0.5, -0.2),ncol=2,frameon=False,fontsize=9)
    ax.add_artist(leg1)

plot_degradation(axs,truth_last)
plt.savefig('experiments/results/figures/paper/deg.pdf')

# Precision
fig, axs = plt.subplots(1,1, figsize=(4.5, 3.2))
plt.subplots_adjust(top=0.93,bottom=0.3,left=0.15,right=0.98, hspace=0.0,wspace=0.2)

def precision(truth_selection,truth_best,min_truth):
    return (truth_selection-min_truth)/(truth_best-min_truth)

def plot_precision(ax,truth_last):

    ax.plot(range(65+1),truth_last[:65+1],color='blue',linewidth=1)
    ax.plot(range(65,100),truth_last[65:101],color='blue',linestyle='--',linewidth=1,alpha=0.3)
    ax.axhline(y=min(truth_last[:65+1]), color='red', linestyle='-',alpha=0.5)
    ax.text(0.65, 0.01,r'$\text{min}_{0,65}$',transform=ax.get_yaxis_transform(),color='red',va='bottom')

    #t=35, politica seleccionada por criterio
    ax.scatter(35, truth_last[35], color='orange')
    ax.text(35-8, truth_last[35]+0.1,r'$\widetilde{\pi}_{65}$', va='center',color='orange')
    ax.hlines(y=truth_last[35], xmin=35, xmax=truth_last[:65+1].index(max(truth_last[:65+1])), color='orange',linestyle='--',linewidth=1)


    ax.scatter(50, truth_last[50], color='green')
    ax.text(50-8, truth_last[50]+0.1,r'$\widetilde{\pi}_{65}$', va='center',color='green')
    ax.hlines(y=truth_last[50], xmin=50, xmax=truth_last.index(max(truth_last)), color='green',linestyle='--',linewidth=1)


    ax.scatter(truth_last.index(max(truth_last[:65+1])), max(truth_last[:65+1]), color='black')
    ax.text(truth_last.index(max(truth_last[:65+1])), max(truth_last[:65+1])+0.1,r'$\pi^*_{65}$', va='center',color='black')
    ax.vlines(x=truth_last.index(max(truth_last[:65+1])), ymin=min(truth_last[:65+1]), ymax=max(truth_last[:65+1]), color='grey',linestyle='-',linewidth=5,alpha=0.3)
    ax.vlines(x=truth_last.index(max(truth_last[:65+1])), ymin=min(truth_last[:65+1]), ymax=truth_last[35], color="orange",linestyle='-',linewidth=5,alpha=0.5)


    ax.scatter(truth_last.index(max(truth_last)), max(truth_last), color='black')
    ax.text(truth_last.index(max(truth_last)), max(truth_last)+0.1,r"$\pi^*_{100}$", va='center',color='black')
    ax.vlines(x=truth_last.index(max(truth_last)), ymin=min(truth_last[:65+1]), ymax=max(truth_last), color='grey',linestyle='-',linewidth=5,alpha=0.3)
    ax.vlines(x=truth_last.index(max(truth_last)), ymin=min(truth_last[:65+1]), ymax=truth_last[50], color="green",linestyle='-',linewidth=5,alpha=0.5)


    prec1=round(precision(truth_last[35],max(truth_last[:65+1]),min(truth_last[:65+1])),2)
    prec2=round(precision(truth_last[50],max(truth_last),min(truth_last)),2)
    ax.text(0.02, 0.98,rf'Train: $\alpha_{{65}}$ = {prec1}',transform=ax.transAxes,ha='left',va='top',fontsize=10,color='orange')
    ax.text(0.02, 0.89,rf'Test: $\alpha_{{65}}$ = {prec2}',transform=ax.transAxes,ha='left',va='top',fontsize=10,color='green')
    ax.text(0.02, 0.80,r'$(t^\prime = 35)$',transform=ax.transAxes,ha='left',va='top',fontsize=10,color='green')

    ax.set_ylabel(r'$f(\pi_t)$',fontsize=10)
    ax.set_title('Accuracy',fontsize=10)
    ax.set_xlabel(r'$t$',fontsize=10)

    ax.set_ylim([-0.1, 1.3])
    ax.set_yticks(np.arange(0, 1.01, 0.25))

    # Leyenda
    legend_handles = [
    mlines.Line2D([], [], color='grey', linestyle='-', linewidth=6, alpha=0.3,label=r'denominator of $\alpha_t$'),
    mlines.Line2D([], [], color='orange', linestyle='-', linewidth=6, alpha=0.5,label=r'numerator of $\alpha_t$ for selection 1'),
    mlines.Line2D([], [], color='green', linestyle='-', linewidth=6, alpha=0.5,label=r'numerator of $\alpha_t$ for selection 2'),
    mlines.Line2D([], [], color='blue', linestyle='--',linewidth=2, label=r"reward evolution by investing validation time $t'$ learning"),
]

    leg1 = ax.legend(handles=legend_handles[:2],loc='upper center',bbox_to_anchor=(0.43, -0.15),ncol=2,frameon=False,fontsize=9,columnspacing=1.5)
    leg2 = ax.legend(handles=legend_handles[2:3],loc='upper center',bbox_to_anchor=(0.43, -0.245),ncol=1,frameon=False,fontsize=9,columnspacing=1.5)
    leg3 = ax.legend(handles=legend_handles[3:],loc='upper center',bbox_to_anchor=(0.43, -0.34),ncol=1,frameon=False,fontsize=9,columnspacing=1.5)
    ax.add_artist(leg1)
    ax.add_artist(leg2)

plot_precision(axs,truth_last)
plt.savefig('experiments/results/figures/paper/acc.pdf')

# Eficiencia
fig, axs = plt.subplots(1,1, figsize=(4.5, 3))
plt.subplots_adjust(top=0.93,bottom=0.3,left=0.15,right=0.98, hspace=0.0,wspace=0.2)

def efficiency(truth_selection,all_truth):
    indice = next(i for i, x in enumerate(all_truth) if x >= truth_selection)
    return indice,(indice+1)/len(all_truth)

def plot_efficiency(ax,truth_last):

    ax.plot(range(len(truth_last)),truth_last,color='blue',linewidth=1)

    ax.axvline(x=100, color='red', linestyle='-',alpha=0.5)

    #t=42, eficiencia baja por criterio
    t_min,eff=efficiency(truth_last[42],truth_last)
    ax.hlines(y=truth_last[42], xmin=0, xmax=100, color="grey",linestyle='-',linewidth=5,alpha=0.3,zorder=1)
    ax.hlines(y=truth_last[42], xmin=0, xmax=t_min, color='orange',linestyle='-',linewidth=5,alpha=0.5,zorder=2)
    ax.scatter(42, truth_last[42], color='orange',zorder=10)
    ax.text(42+3, truth_last[42]-0.2,r'$\widetilde{\pi}_{100}$',color='orange',va='bottom')
    ax.scatter(t_min, truth_last[42], color='black',marker='x',zorder=10)
    ax.text(0.05, truth_last[42]+0.03,r'$\varepsilon_{100}=$'+str(round(eff,2)),transform=ax.get_yaxis_transform(),color='orange',va='bottom')


    #t=92, eficiencia baja por convergencia
    t_min,eff=efficiency(truth_last[92],truth_last)
    ax.hlines(y=truth_last[92], xmin=0, xmax=100, color="grey",linestyle='-',linewidth=5,alpha=0.3,zorder=1)
    ax.hlines(y=truth_last[92], xmin=0, xmax=t_min, color='green',linestyle='-',linewidth=5,alpha=0.5,zorder=2)
    ax.scatter(92, truth_last[92], color='green',zorder=10)
    ax.text(92-2, truth_last[92]-0.3,r'$\widetilde{\pi}_{100}$',color='green',va='bottom')
    ax.scatter(t_min, truth_last[92], color='black',zorder=10,marker='x')
    ax.text(0.05, truth_last[92]-0.15,r'$\varepsilon_{100}=$'+str(round(eff,2)),transform=ax.get_yaxis_transform(),color='green',va='bottom')

    ax.set_ylabel(r'$f(\pi_t)$',fontsize=10)
    ax.set_title('Efficiency',fontsize=10)
    ax.set_xlabel(r'$t$',fontsize=10)

    # Leyenda
    legend_handles = [
    mlines.Line2D([], [], color='grey', linestyle='-', linewidth=6, alpha=0.3,label=r'denominator of $\varepsilon_t$ '),
    mlines.Line2D([], [], color='orange', linestyle='-', linewidth=6, alpha=0.5,label=r'numerator of $\varepsilon_t$ for selection 1'),
    mlines.Line2D([], [], color='green', linestyle='-', linewidth=6, alpha=0.5,label=r'numerator of $\varepsilon_t$ for selection 2'),
    mlines.Line2D([], [], color='black', marker='x', linestyle='None',markersize=7, label='first time with same reward'),]

    leg1 = ax.legend(handles=legend_handles[:2],loc='upper center',bbox_to_anchor=(0.43, -0.2),ncol=2,frameon=False,fontsize=9,columnspacing=1.5)
    leg2 = ax.legend(handles=legend_handles[2:],loc='upper center',bbox_to_anchor=(0.43, -0.3),ncol=2,frameon=False,fontsize=9,columnspacing=0.5)
    ax.add_artist(leg1)

plot_efficiency(axs,truth_last)
plt.savefig('experiments/results/figures/paper/eff.pdf')





