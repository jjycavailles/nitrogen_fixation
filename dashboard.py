import nbinteract as nbi
import ipywidgets as widgets
from ipywidgets import Layout
from io import BytesIO
import colorsys
import pickle # package use to save data
from IPython.display import display, Image
import numpy as np

from main import *

class Select_box_new(widgets.VBox):
    def __init__(self, dashboard):
        import numpy as np
        self.dashboard = dashboard
        
        self.selection_new = widgets.Button(
        description='new randomization',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me to simulate again dynamical systems with a different randomization',
        icon='check' # (FontAwesome names without the `fa-` prefix)
        )
        
        self.selection_new.on_click(dashboard.on_new_selected)
        
        children = [
        self.selection_new,
        ]
        super().__init__(children)#, layout=Layout(width="100%"))   
        
        
class Select_box_finalfime(widgets.VBox):
    def __init__(self, dashboard):
        
        import numpy as np
        
        self.dashboard = dashboard

        self.selection_finaltime = widgets.FloatLogSlider(min = np.log10(10), max = np.log10(10**5), description = "final time", value = 10**3)
        self.selection_finaltime.observe(dashboard.on_finalfime_selected, names = "value")
        
        
        children = [
        self.selection_finaltime,
        ]
        super().__init__(children)#, layout=Layout(width="100%"))  

class Select_box_tropic(widgets.VBox):
    def __init__(self, dashboard):
        
        import numpy as np
        
        self.dashboard = dashboard


        #initial point
        self.selection_non0_t= widgets.FloatSlider(min = 0, max = 1, description = "non fixers t=0", value = 0.45)
        self.selection_non0_t.observe(dashboard.on_non0_t_selected, names = "value")
        
        self.selection_fac0_t= widgets.FloatSlider(min = 0, max = 1, description = "facultative t=0", value = 0.45)
        self.selection_fac0_t.observe(dashboard.on_fac0_t_selected, names = "value")
        
        self.selection_obl0_t= widgets.FloatSlider(min = 0, max = 1, description = "obligate t=0", value = 0.1)
        self.selection_obl0_t.observe(dashboard.on_obl0_t_selected, names = "value")
        
        
        self.selection_param = widgets.Dropdown(options=['observable quantity', 'transition'], value='transition', description='Parameter', disabled=True)
        self.selection_param.observe(self.on_param_selected, names = "value")        
        
        #self.on_param_selected()
        self.selection_param1 = widgets.FloatLogSlider(min = np.log10(0.005), max = np.log10(0.5), description = "plh", value = 0.05)
        self.selection_param1.observe(self.dashboard.on_param1_t_selected, names = "value")
        
        self.selection_param2 = widgets.FloatLogSlider(min = np.log10(0.002), max = np.log10(0.2), description = "phl", value = 0.02)
        self.selection_param2.observe(self.dashboard.on_param2_t_selected, names = "value")
        
        children = [
        self.selection_param,
#        self.selection_phl_t,
#        self.selection_plh_t,
        self.selection_param1,
        self.selection_param2,
        self.selection_non0_t,
        self.selection_fac0_t,
        self.selection_obl0_t,
        ]
        super().__init__(children)#, layout=Layout(width="100%"))   
        

    def on_param_selected(self, change):
        self.dashboard.on_param_selected(change)
            #     """
        if(change["new"] == 'observable quantity'):
            self.selection_param1 = widgets.FloatLogSlider(min = np.log10(1./700), max = np.log10(1./7), description = "disturbance\nfrequency", value = 1./70, disable = True)
            #self.selection_param1.observe(self.on_freq_selected, names = "value")
            self.selection_param2 = widgets.FloatLogSlider(min = np.log10(10), max = np.log10(40), description = "time\nrecovery", value = 20)
            #self.selection_param2.observe(self.on_recovery_selected, names = "value")
            
            
        elif(change["new"] == 'transition'):
            self.selection_param1 = widgets.FloatLogSlider(min = np.log10(0.005), max = np.log10(0.5), description = "plh", value = 0.05)
            self.selection_param2 = widgets.FloatLogSlider(min = np.log10(0.002), max = np.log10(0.2), description = "phl", value = 0.02)
            
            """
            self.selection_plh_t = widgets.FloatLogSlider(min = np.log10(0.005), max = np.log10(0.5), description = "plh", value = 0.05)
            self.selection_plh_t.observe(self.dashboard.on_plh_t_selected, names = "value")
            
            self.selection_phl_t = widgets.FloatLogSlider(min = np.log10(0.002), max = np.log10(0.2), description = "phl", value = 0.02)
            self.selection_phl_t.observe(self.dashboard.on_phl_t_selected, names = "value")
            """    
        else:
            print("Wrong choice of parameter type (for transition)")
        
        
        self.selection_param1.observe(self.dashboard.on_param1_t_selected, names = "value")
        self.selection_param2.observe(self.dashboard.on_param2_t_selected, names = "value")
        
        
        children = [
        self.selection_non0_t,
        self.selection_fac0_t,
        self.selection_obl0_t,
        self.selection_param,
#        self.selection_phl_t,
#        self.selection_plh_t,
        self.selection_param1,
        self.selection_param2,
        ]
        super().__init__(children)#, layout=Layout(width="100%"))  
    #    """
class Select_box_extra(widgets.VBox):
    def __init__(self, dashboard):
        
        import numpy as np
        
        self.dashboard = dashboard


        #initial point
        self.selection_non0_e = widgets.FloatSlider(min = 0, max = 1, description = "non fixers t=0", value = 0.45)
        self.selection_non0_e.observe(dashboard.on_non0_e_selected, names = "value")
        
        self.selection_fac0_e= widgets.FloatSlider(min = 0, max = 1, description = "facultative t=0", value = 0.1)
        self.selection_fac0_e.observe(dashboard.on_fac0_e_selected, names = "value")
        
        self.selection_obl0_e= widgets.FloatSlider(min = 0, max = 1, description = "obligate t=0", value = 0.45)
        self.selection_obl0_e.observe(dashboard.on_obl0_e_selected, names = "value")
        
        self.selection_param = widgets.Dropdown(options=['observable quantity', 'transition'], value='transition', description='Parameter', disabled=True)
        #self.selection_param.observe(self.on_param_selected, names = "value")

        self.selection_plh_e = widgets.FloatLogSlider(min = np.log10(0.005), max = np.log10(0.5), description = "plh", value = 0.02)
        self.selection_plh_e.observe(dashboard.on_plh_e_selected, names = "value")
        
        self.selection_phl_e = widgets.FloatLogSlider(min = np.log10(0.002), max = np.log10(0.2), description = "phl", value = 0.01)
        self.selection_phl_e.observe(dashboard.on_phl_e_selected, names = "value")
        
        
        children = [
        self.selection_param,
        self.selection_phl_e,
        self.selection_plh_e,
        self.selection_non0_e,
        self.selection_fac0_e,
        self.selection_obl0_e,
        ]
        super().__init__(children)#, layout=Layout(width="100%"))   
        
class transition_box(widgets.Box):
    def __init__(self, dashboard, forest):
        #%pip install matplotlib
        #import matplotlib.pyplot as plt
        self.html = widgets.HTML()
        self.dashboard = dashboard
        
        self.forest = forest
        
        if(self.forest =="tropical"):
            self.phl = self.dashboard.phl_t
            self.plh = self.dashboard.plh_t
        elif(self.forest == "extra"):
            self.phl = self.dashboard.phl_e
            self.plh = self.dashboard.plh_e
        else:
            print("wrong choice of forest type")
        
        self.print_html()
        
        html_container = widgets.Box([self.html], layout=Layout(width="100%"))
        
        children = [
        self.html,   
        ]
        super().__init__(children)#, layout=Layout(width="100%"))
        
    def print_html(self):
        self.phl
        self.plh
        pl = self.phl/(self.phl+self.plh)
        ph = self.plh/(self.phl+self.plh)
        rec = (1-self.plh)/self.plh
        freq = 1./(1./self.phl + rec -1)
        nbre_dec = 5
        self.html.value = "pl = "+str(round(pl, nbre_dec))+"<br />ph = "+str(round(ph, nbre_dec))+"<br />recovery time = "+str(round(rec, nbre_dec))+"<br />disturbance frequency = "+str(round(freq, nbre_dec))
        return
        

    
    def change_phl(self, change):
        self.phl = change
        self.print_html()
        return
    
    def change_plh(self, change):
        self.plh = change
        self.print_html()
        return
 

class Select_box_low(widgets.VBox):
    def __init__(self, dashboard):
        
        import numpy as np
        
        self.dashboard = dashboard

        self.selection_alpha = widgets.FloatLogSlider(min = np.log10(0.0001), max = np.log10(0.01), description = "alpha", value = 0.001, tooltip = "alpha have to lower than beta")
        self.selection_alpha.observe(dashboard.on_alpha_selected, names = "value")
        
        self.selection_beta = widgets.FloatLogSlider(min = np.log10(0.0008), max = np.log10(0.08), description = "beta", value = 0.008)
        self.selection_beta.observe(dashboard.on_beta_selected, names = "value")

        self.selection_gamma = widgets.FloatLogSlider(min = np.log10(0.00036), max = np.log10(0.036), description = "gamma", value = 0.0036)
        self.selection_gamma.observe(dashboard.on_gamma_selected, names = "value")

#        self.selection_tfinal = widgets.FloatLogSlider(min = np.log10(10), max = np.log10(10**5), description = "Final time", value = 10000)
#        self.selection_tfinal.observe(dashboard.tfinal, names = "value")
                        
        children = [
        self.selection_alpha,
        self.selection_beta,
        self.selection_gamma,
        ]
        super().__init__(children, layout=Layout(width="100%"))   
        
class Select_box_high(widgets.VBox):
    def __init__(self, dashboard):
        
        import numpy as np
        
        self.dashboard = dashboard

        self.selection_A = widgets.FloatLogSlider(min = np.log10(0.000036), max = np.log10(0.0036), description = "A", value = 0.00036)
        self.selection_A.observe(dashboard.on_A_selected, names = "value")

        self.selection_B = widgets.FloatLogSlider(min = np.log10(0.0005), max = np.log10(0.05), description = "B", value = 0.005)
        self.selection_B.observe(dashboard.on_B_selected, names = "value")

        self.selection_Gamma = widgets.FloatLogSlider(min = np.log10(0.00016), max = np.log10(0.016), description = "Gamma", value = 0.0016)
        self.selection_Gamma.observe(dashboard.on_Gamma_selected, names = "value")

 #       self.selection_tfinal = widgets.FloatLogSlider(min = np.log10(10), max = np.log10(10**5), description = "Final time", value = 10000)
 #       self.selection_tfinal.observe(dashboard.tfinal, names = "value")
                        
        children = [
        self.selection_A,
        self.selection_B,
        self.selection_Gamma,
        ]
        super().__init__(children, layout=Layout(width="100%"))   
        
        
class Image_box(widgets.Box):
    def __init__(self, dashboard, forest):
        #%pip install matplotlib
        #import matplotlib.pyplot as plt
        self.image = widgets.Image()
        self.dashboard = dashboard
        
        
        self.forest = forest
        if(self.forest == "tropical"):
            self.phl = self.dashboard.phl_t
            self.plh = self.dashboard.plh_t
            self.non0 = self.dashboard.non0_t
            self.fac0 = self.dashboard.fac0_t
            self.obl0 = self.dashboard.obl0_t
            
        elif(self.forest == "extra"):
            self.phl = self.dashboard.phl_e
            self.plh = self.dashboard.plh_e
            self.non0 = self.dashboard.non0_e
            self.fac0 = self.dashboard.fac0_e
            self.obl0 = self.dashboard.obl0_e
            
        else:
            print("Forest type unknown")
            
        self.alpha = self.dashboard.alpha
        self.beta = self.dashboard.beta
        self.gamma = self.dashboard.gamma
        
        self.A = self.dashboard.A
        self.B = self.dashboard.B
        self.Gamma = self.dashboard.Gamma
        
        self.finaltime = self.dashboard.finaltime
        
        
        self.print_image()
        
        image_container = widgets.Box([self.image], layout=Layout(width="100%"))
        
        children = [
   #     image_container,
        self.image,   
        ]
        super().__init__(children)#, layout=Layout(width="100%"))
        
    def print_image(self):
        #"""
        try:
            import matplotlib.pyplot as plt
        except:
            #pip install matplotlib -qqq
            import matplotlib.pyplot as plt
        #"""
        #import main
        #import matplotlib.pyplot as plt
        #exec(open("main.py").read(), globals())
        
        dt_p = 0.1/max([self.plh, self.phl]) # to have probability < 1
        dt_d = min([10, self.finaltime/100])
        dt = min([dt_p, dt_d])
        T = np.arange(0, int(self.finaltime), dt)
        
        #T = np.linspace(1, int(self.finaltime), max([self.finaltime//10 + 1, 100]))
        A = payoff_matrix_param(self.alpha, self.beta, self.gamma, self.A, self.B, self.Gamma)
        #print("proba", self.plh*(T[1]-T[0]))
        #print("proba", self.phl*(T[1]-T[0]))
        #print("A\n", A)
        #XXe, NNe = solve(T, A, plh = self.plh_e, phl = self.phl_e, X0 = [self.non0, self.fac0, self.obl0]) 
        #XXt, NNt = solve(T, A, plh = self.plh_t, phl = self.phl_t, X0 = [self.non0, self.fac0, self.obl0]) 
        XX, NN = solve(T, A, plh = self.plh, phl = self.phl, X0 = [self.non0, self.fac0, self.obl0]) 
        
       # fig = plt.figure(figsize=(16, 6))
        plt.subplots(2,1, figsize = (9, 5)) 
        plt.subplot(2,1,1)
        
        if(self.forest == "tropical"):
            plot_biomass(T, XX, NN, "Tropical") 
            plt.ylabel("Biomass proportion", fontsize = 12)
       # plt.subplot(2,1,2)
       # plot_biomass(T, XXe, NNe, "Extra tropical")
        elif(self.forest == "extra"):
            plot_biomass(T, XX, NN, "Extra") 
            plt.ylabel("")
        plt.xlabel("")
        #plt.show()
        

#        plt.subplots(1,2, figsize = (20, 1)) 
        plt.subplot(2,1,2) 
        plot_nitrogen(T, XX, NN, self.forest)
        if(self.forest == "tropical"):
            plt.ylabel("Nitrogen", fontsize = 12)
        else:
            plt.ylabel("", fontsize = 12)
        plt.title("")
        plt.yticks([0,1], ["low", "high"])
        plt.xlabel("Time", fontsize = 12)
            
        """
        plt.subplot(2,2,4)
        plot_nitrogen(T, XXe, NNe, "Extra tropical")
        plt.title("")
        plt.yticks([0,1], ["low", "high"])
        plt.ylabel("", fontsize = 12)
        plt.xlabel("Time", fontsize = 12)
        #plt.show()    
        """
        
        """
        D = Dynamic(h=h, l=l, alpha=alpha, c=c, model=self.model, tFinal=50)
        D.initialisation()
        if(self.plot == "time series"):
            D.eulerEx()
            D.plot(show=False)
            plt.legend(fontsize = 20) # duplicate ...
        elif(self.plot == "phase portrait"):
            D.plot_phase_portrait(show=False)
        elif(self.plot == "histogram"):
            D.plot_histogram(show=False)
        """
            
        image_file = BytesIO()
        #fig.savefig(fname = image_file)
        plt.savefig(fname = image_file)
        image_file.seek(0)
        image_data = image_file.read()
        self.image.value = image_data
#            self.image.width = 1500
#           self.image.height = 2000
        plt.close()

#          file = open(nom, "rb")
#         image = file.read()
        #plt.imshow(image)
#        self.image.value = image
 #       self.image.format = 'png'
        
        
        return
        

    def change_non0(self, change):
        self.non0 = change
        self.print_image()
        return
    
    def change_fac0(self, change):
        self.fac0 = change
        self.print_image()
        return
    
    def change_obl0(self, change):
        self.obl0 = change
        self.print_image()
        return
    
    def change_plh(self, change):
        self.plh = change
        self.print_image()
        return
    
    
    def change_phl(self, change):
        self.phl = change
        self.print_image()
        return
    
    
    
    def change_alpha(self, change):
        self.alpha = change
        self.print_image()
        return
    
    def change_beta(self, change):
        self.beta = change
        self.print_image()
        return
    
    def change_gamma(self, change):
        self.gamma = change
        self.print_image()
        return
    
    
    def change_A(self, change):
        self.A = change
        self.print_image()
        return
    
    def change_B(self, change):
        self.B = change
        self.print_image()
        return
    
    def change_Gamma(self, change):
        self.Gamma = change
        self.print_image()
        return
    
    def change_finaltime(self, change):
        self.finaltime = change
        self.print_image()
        return
    
class Image_box_dominant(widgets.Box):
    def __init__(self, dashboard):
        #%pip install matplotlib
        #import matplotlib.pyplot as plt
        self.image = widgets.Image()
        self.dashboard = dashboard
        
        self.phl_t = self.dashboard.phl_t
        self.plh_t = self.dashboard.plh_t
        self.phl_e = self.dashboard.phl_e
        self.plh_e = self.dashboard.plh_e

        self.alpha = self.dashboard.alpha
        self.beta = self.dashboard.beta
        self.gamma = self.dashboard.gamma
        
        self.A = self.dashboard.A
        self.B = self.dashboard.B
        self.Gamma = self.dashboard.Gamma
        
  #      self.finaltime = self.dashboard.finaltime
        
        
        self.print_image()
        
        image_container = widgets.Box([self.image], layout=Layout(width="100%"))
        
        children = [
   #     image_container,
        self.image,   
        ]
        super().__init__(children)#, layout=Layout(width="100%"))
        
    def print_image(self):
        """
        try:
            import matplotlib.pyplot as plt
        except:
            %pip install matplotlib -qqq
            import matplotlib.pyplot as plt
        """
        
        
        pl_t = self.phl_t / (self.phl_t + self.plh_t)
        pl_e = self.phl_e / (self.phl_e + self.plh_e)
        
        """
        A = 0.05
        alpha = 0.001
        B = 0.02
        beta = 0.01
        Gamma = 0.0001
        gamma = 0.0002
        """
        
        plt.figure(figsize = (10, 2))
        plt.plot([0,1], [0,0], "black")
        plt.xlim(0, 1)
        #plt.xticks([0, pl_t, pl_e, 1])
        #plt.xticks([0, 1])
        plt.xticks([])
        plt.ylim(-2, 3)
        plt.yticks([])
        plt.box(on=None)
        #plt.text(0, 1.5, 'boxed italics text in data coords', style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
        plt.ylabel("dominant\nstrategies", fontsize = 15)
        #plt.xlabel("pl extra", fontsize = 15, rotation = 90)
        plt.annotate('  pl extra = '+str(round(pl_e, 6)), xy=(pl_e, -0.), xytext=(pl_e, -.5), arrowprops=dict(facecolor='black', shrink=0.8))
        plt.annotate('pl tropic = '+str(round(pl_t, 6)), xy=(pl_t, -0.), xytext=(pl_t, -1.5), arrowprops=dict(facecolor='black', shrink=0.1))
        #plt.axis.set_visible(False)
        
        limit_nf = self.A/(self.A+self.alpha)
        limit_no = self.B/(self.B+self.beta)
        limit_fo = self.Gamma/(self.gamma+self.Gamma)
        
        plt.fill_betweenx(y=[0,10], x1=0, x2 = min([limit_nf, limit_no]), label = "non fixers")
        plt.fill_betweenx(y=[0,10], x1=limit_nf, x2 = limit_fo, where = [limit_nf< limit_fo, limit_nf< limit_fo], label = "facultative fixers")
        plt.fill_betweenx(y=[0,10], x1=max([limit_fo, limit_no]), x2 = 1, label = "obligate fixers")
        plt.legend()

        image_file = BytesIO()
        #fig.savefig(fname = image_file)
        plt.savefig(fname = image_file)
        image_file.seek(0)
        image_data = image_file.read()
        self.image.value = image_data
#            self.image.width = 1500
#           self.image.height = 2000
        plt.close()
#          file = open(nom, "rb")
#         image = file.read()
        #plt.imshow(image)
#        self.image.value = image
 #       self.image.format = 'png'
        return
        

        
        
    
    def change_plh_t(self, change):
        self.plh_t = change
        self.print_image()
        return
    
    
    def change_phl_t(self, change):
        self.phl_t = change
        self.print_image()
        return
    
    
    def change_plh_e(self, change):
        self.plh_e = change
        self.print_image()
        return
    
    
    def change_phl_e(self, change):
        self.phl_e = change
        self.print_image()
        return
    
    
    
    def change_alpha(self, change):
        self.alpha = change
        self.print_image()
        return
    
    def change_beta(self, change):
        self.beta = change
        self.print_image()
        return
    
    def change_gamma(self, change):
        self.gamma = change
        self.print_image()
        return
    
    
    def change_A(self, change):
        self.A = change
        self.print_image()
        return
    
    def change_B(self, change):
        self.B = change
        self.print_image()
        return
    
    def change_Gamma(self, change):
        self.Gamma = change
        self.print_image()
        return
    
class Image2_box(widgets.Box):
    def __init__(self, dashboard, level):
        #%pip install matplotlib
        #import matplotlib.pyplot as plt
        self.image = widgets.Image()
        self.dashboard = dashboard
        
        """
        self.non0 = self.dashboard.non0
        self.fac0 = self.dashboard.fac0
        self.obl0 = self.dashboard.obl0
        
        self.phl_t = self.dashboard.phl_t
        self.plh_t = self.dashboard.plh_t
        self.phl_e = self.dashboard.phl_e
        self.plh_e = self.dashboard.plh_e
  
        """
        self.level = level
        #if(self.level=="low"):
        self.alpha = self.dashboard.alpha
        self.beta = self.dashboard.beta
        self.gamma = self.dashboard.gamma
        #elif(self.level=="high"):
        self.A = self.dashboard.A
        self.B = self.dashboard.B
        self.Gamma = self.dashboard.Gamma
        #else:
        #    print("Wrong choice of nitrogen level")
        
        self.finaltime = self.dashboard.finaltime
        
        self.print_image()
        
        image_container = widgets.Box([self.image], layout=Layout(width="100%"))
        
        children = [
   #     image_container,
        self.image,   
        ]
        super().__init__(children, layout=Layout(width="100%"))
        
    def print_image(self):
        #"""
        try:
            import matplotlib.pyplot as plt
        except:
            #%pip install matplotlib -qqq
            import matplotlib.pyplot as plt
        #"""
        #import main
        #import matplotlib.pyplot as plt
        #exec(open("main.py").read(), globals())
 #       A = payoff_matrix_param(self.alpha, self.beta, self.gamma, self.A, self.B, self.Gamma)
      
        if(self.level=="low"):
            An = payoff_matrix_param_low(self.alpha, self.beta, self.gamma)
        if(self.level=="high"):
            An = payoff_matrix_param_high(self.A, self.B, self.Gamma)

        #plt.subplots(1,2, figsize = (16, 5))
        #plt.figure(figsize=(8,5))
        plt.figure(figsize=(6,3))
        #Al = A[0]
        T = np.linspace(1, self.finaltime, max([self.finaltime//10 + 1, 100]))
        plot_determinist_level(An, T, show=False)
        plt.title("Nitrogen "+self.level, fontsize=25)
            
        image_file = BytesIO()
        #fig.savefig(fname = image_file)
        plt.savefig(fname = image_file)
        image_file.seek(0)
        image_data = image_file.read()
        self.image.value = image_data
#            self.image.width = 1500
#           self.image.height = 2000
        plt.close()
        return
        
    """
    def change_non0(self, change):
        self.non0 = change
        self.print_image()
        return
    
    def change_fac0(self, change):
        self.fac0 = change
        self.print_image()
        return
    
    def change_obl0(self, change):
        self.obl0 = change
        self.print_image()
        return
    """
    
    def change_a(self, change):
        if(self.level=="low"):
            self.alpha = change
        else:
            self.A = change
        self.print_image()
        return
    
    def change_b(self, change):
        if(self.level=="low"):
            self.beta = change
        else:
            self.B = change
        self.print_image()
        return
    
    def change_c(self, change):
        if(self.level=="low"):
            self.gamma = change
        else:
            self.Gamma = change
        self.print_image()
        return
    
    def change_finaltime(self, change):
        self.finaltime = change
        self.print_image()
        return
    
class matrix_box(widgets.Box):
    def __init__(self, dashboard, level):
        #%pip install matplotlib
        #import matplotlib.pyplot as plt
        self.html = widgets.HTML()
        self.dashboard = dashboard
        
        self.level = level
        if(self.level =="low"):
            self.a = self.dashboard.alpha
            self.b = self.dashboard.beta
            self.c = self.dashboard.gamma
        elif(self.level == "high"):
            self.a = - self.dashboard.A
            self.b = - self.dashboard.B
            self.c = - self.dashboard.Gamma
        else:
            print("wrong choice of nitrogen level")
        
        self.print_html()
        
        html_container = widgets.Box([self.html], layout=Layout(width="100%"))
        
        children = [
        self.html,   
        ]
        super().__init__(children, layout=Layout(self_content='center'))
        
    def print_html(self):
        nbre_units = 6
        a = round(self.a, nbre_units)
        b = round(self.b, nbre_units)
        c = round(self.c, nbre_units)
        self.html.value = "<style> table, th, td {  border: 1px solid black;  border-collapse: collapse;}th, td {  padding: 5px;  text-align: left;} caption { color:black ;  padding: 10px ; text-align: center;}</style>  <table> <caption> Nitrogen "+self.level+" </caption> <tr>    <th> </th> <th>non fixers</th>    <th>facultative fixers</th>    <th>obligate fixers</th>  </tr>  <tr>  <th> non fixers </th>  <td>0</td>    <td>"+str(-a)+"</td>    <td>"+str(-b)+"</td>  </tr>  <tr> <th>facultative fixers</th>   <td>"+str(a)+"</td>    <td>0</td>    <td>"+str(-c)+"</td>  </tr>  <tr>  <th>obligate fixers</th>  <td>"+str(b)+"</td>    <td>"+str(c)+"</td>    <td>0</td>  </tr> </table>"
        return
        

    
    def change_a(self, change):
        if(self.level=="low"):
            self.a = change
        else:
            self.a = - change
        self.print_html()
        return
    
    def change_b(self, change):
        if(self.level=="low"):
            self.b = change
        else:
            self.b = - change
        self.print_html()
        return
    
    def change_c(self, change):
        if(self.level=="low"):
            self.c = change
        else:
            self.c = - change
        self.print_html()
        return
    
class Dashboard(widgets.VBox):
    def __init__(self):

        self.non0_t = 0.45
        self.fac0_t = 0.45
        self.obl0_t = 0.1
        self.non0_e = 0.45
        self.fac0_e = 0.1
        self.obl0_e = 0.45
        
        
        self.phl_t = 0.05
        self.plh_t = 0.02
        self.phl_e = 0.02
        self.plh_e = 0.01
        
        #"""
        self.param1_t = 0.05
        self.param2_t = 0.02
        self.param1_e = 0.02
        self.param2_e = 0.01
        #"""
        
        # [0.001, 0.008, 0.0036, 0.00036, 0.005, 0.0016]
        self.alpha = 0.001
        self.beta = 0.008
        self.gamma = 0.0036
        self.A = 0.00036
        self.B = 0.005
        self.Gamma = 0.0016
        
        self.finaltime = 10**4
        
        self.param = "transition"
        
        #self.visualization = ['tropical', 'extra', 'low', 'high', 'matrices']
    
        self.select_box_tropic = Select_box_tropic(self)
        self.select_box_extra = Select_box_extra(self)
        
        self.transition_box_tropical = transition_box(self, "tropical")
        self.transition_box_extra = transition_box(self, "extra")
        
        self.select_box_low = Select_box_low(self)
        self.select_box_high = Select_box_high(self)
        
        self.select_finaltime = Select_box_finalfime(self)
        #self.select_visualization = Select_box_visualization(self)
        
    #    self.text_box = Text_box(self)
        self.image_box_tropical = Image_box(self, "tropical")
        self.image_box_extra = Image_box(self, "extra")
        
        self.image_box_dominant = Image_box_dominant(self)
        
        self.image2_box_low = Image2_box(self, level = "low")
        self.image2_box_high = Image2_box(self, level = "high")
        
        self.matrix_box_l = matrix_box(self, "low")
        self.matrix_box_h = matrix_box(self, "high")
        
        self.new = Select_box_new(self)
        
        
        
        row_text = widgets.Box([self.transition_box_tropical, self.transition_box_extra], layout=Layout(width="100%"))
        
        row0 = widgets.Box([self.select_box_tropic,self.select_box_extra, self.select_box_low, self.select_box_high, self.select_finaltime], layout=Layout(width="100%"))
        C1 = widgets.Box([self.image_box_tropical, self.image_box_extra], layout=Layout(width="100%", align_self = 'center'))
        #C2 = widgets.Box([self.select_box], layout=Layout(width="32%"))
        
        C01 = widgets.Box([self.image2_box_low, self.image2_box_high], layout=Layout(width="100%", align_self = 'center'))
        #rowA = widgets.Box([self.image_box, self.select_box], layout=Layout(width="100%"))
        #rowA = widgets.Box([C1, C2], layout=Layout(width="100%"))
        rowA = widgets.Box([C1], layout=Layout(width="100%"))
        rowB = widgets.Box([C01], layout=Layout(width="100%"))
        rowC = widgets.Box([self.matrix_box_l, self.matrix_box_h], layout=Layout(width="100%"))
        #super().__init__([self.new, row0, row_text, rowA, self.image_box_dominant, rowB, rowC], layout=Layout(width="100%"))
        
        
#        col_tropic = widgets.VBox([self.select_box_tropic, self.transition_box_tropical], layout=Layout(width="100%", align_content = 'center')) 
 #       col_extra = widgets.VBox([self.select_box_extra, self.transition_box_extra], layout=Layout(width="100%", align_content = 'center')) 
        
        #row_numeric = widgets.Box([self.new, self.select_finaltime], layout=Layout(width="100%")) 
        #row_forest = widgets.Box([col_tropic, self.image_box_tropical, col_extra, self.image_box_extra], layout=Layout(width="100%"))
        row_select_text_forest = widgets.Box([self.select_box_tropic, self.transition_box_tropical, self.select_box_extra, self.transition_box_extra], layout=Layout(justify_content='center'))
        row_plot_forest = widgets.Box([self.image_box_tropical, self.image_box_extra], layout=Layout(justify_content='center')) 
        row_dominant = widgets.Box([self.image_box_dominant], layout=Layout(justify_content='center'))
        row_low_high = widgets.Box([self.select_box_low, self.image2_box_low, self.select_box_high, self.image2_box_high], layout=Layout(justify_content='center'))
        matrices_low = widgets.Box([self.matrix_box_l], layout=Layout(width="50%", justify_content='center'))
        matrices_high = widgets.Box([self.matrix_box_h], layout=Layout(width="50%", justify_content='center'))
        row_matrices = widgets.Box([matrices_low, matrices_high], layout=Layout(justify_content='center', width="100%"))
        #super().__init__([row_numeric, row_forest, row_dominant, row_low_high, row_matrices], layout=Layout(width="100%"))
        row_finaltime = widgets.Box([self.select_finaltime], layout=Layout(justify_content = 'center'))
        row_new = widgets.Box([self.new], layout=Layout(justify_content='center'))
        super().__init__([row_matrices, row_low_high, row_dominant, row_select_text_forest, row_finaltime, row_new, row_plot_forest])#, layout=Layout(width="100%", justify_content = "center"))
    
    
    def on_non0_t_selected(self, change):
        self.non0_t = change["new"]
        self.image_box_tropical.change_non0(change["new"])
#        self.image2_box.change_non0(change["new"])

    def on_fac0_t_selected(self, change):
        self.fac0_t = change["new"]
        self.image_box_tropical.change_fac0(change["new"])
#        self.image2_box.change_fac0(change["new"])

    def on_obl0_t_selected(self, change):
        self.obl0_t = change["new"]
        self.image_box_tropical.change_obl0(change["new"])
#        self.image2_box.change_obl0(change["new"])

    def on_non0_e_selected(self, change):
        self.non0_e = change["new"]
        self.image_box_extra.change_non0(change["new"])
#        self.image2_box.change_non0(change["new"])

    def on_fac0_e_selected(self, change):
        self.fac0_e = change["new"]
        self.image_box_extra.change_fac0(change["new"])
#        self.image2_box.change_fac0(change["new"])

    def on_obl0_e_selected(self, change):
        self.obl0_e = change["new"]
        self.image_box_extra.change_obl0(change["new"])
#        self.image2_box.change_obl0(change["new"])
    
    
    def on_param_selected(self, change):
        self.param = change["new"]
    
    
    def on_param1_t_selected(self, change):
        self.param1_t = change["new"]
        if(self.param == "transition"):
            self.on_plh_t_selected(change)
            self.plh_t = self.param1_t
        else:
            freq = self.param1_t
            rec = self.param2_t
            plh = 1./freq
            phl = 1./(1+1./freq-rec)
            self.on_plh_t_selected({"new": plh})
            self.on_phl_t_selected({"new": phl})
            self.plh_t = plh
            self.phl_t = phl
            
    def on_param2_t_selected(self, change):
        self.param2_t = change["new"]
        if(self.param == "transition"):
            self.on_phl_t_selected(change)
            self.phl_t = self.param2_t
        else:
            freq = self.param1_t
            rec = self.param2_t
            plh = 1./freq
            phl = 1./(1+1./freq-rec)
            self.on_plh_t_selected({"new": plh})
            self.on_phl_t_selected({"new": phl})
            self.plh_t = plh
            self.phl_t = phl
    #"""
    
    def on_plh_t_selected(self, change):
        self.plh_t = change["new"]
        self.image_box_tropical.change_plh(change["new"])
        self.image_box_dominant.change_plh_t(change["new"])
        self.transition_box_tropical.change_plh(change["new"])
#        self.image2_box.change_plh_t(change["new"])

    def on_phl_t_selected(self, change):
        self.phl_t = change["new"]
        self.image_box_tropical.change_phl(change["new"])
        self.image_box_dominant.change_phl_t(change["new"])
        self.transition_box_tropical.change_phl(change["new"])
#        self.image2_box.change_phl_t(change["new"])

    def on_plh_e_selected(self, change):
        self.plh_e = change["new"]
        self.image_box_extra.change_plh(change["new"])
        self.image_box_dominant.change_plh_e(change["new"])
        self.transition_box_extra.change_plh(change["new"])
#        self.image2_box.change_plh_e(change["new"])

    def on_phl_e_selected(self, change):
        self.phl_e = change["new"]
        self.image_box_extra.change_phl(change["new"])
        self.image_box_dominant.change_phl_e(change["new"])
        self.transition_box_extra.change_phl(change["new"])
#        self.image2_box.change_phl_e(change["new"])

    def on_alpha_selected(self, change):
        self.alpha = change["new"]
        self.matrix_box_l.change_a(change["new"])
        self.image_box_tropical.change_alpha(change["new"])
        self.image_box_extra.change_alpha(change["new"])
        self.image_box_dominant.change_alpha(change["new"])
        self.image2_box_low.change_a(change["new"])
        

    def on_beta_selected(self, change):
        self.beta = change["new"]
        self.matrix_box_l.change_b(change["new"])
        self.image_box_tropical.change_beta(change["new"])
        self.image_box_extra.change_beta(change["new"])
        self.image_box_dominant.change_beta(change["new"])
        self.image2_box_low.change_b(change["new"])

    def on_gamma_selected(self, change):
        self.gamma = change["new"]
        self.matrix_box_l.change_c(change["new"])
        self.image_box_tropical.change_gamma(change["new"])
        self.image_box_extra.change_gamma(change["new"])
        self.image_box_dominant.change_gamma(change["new"])
        self.image2_box_low.change_c(change["new"])

    def on_A_selected(self, change):
        self.A = change["new"]
        self.matrix_box_h.change_a(change["new"])
        self.image_box_tropical.change_A(change["new"])
        self.image_box_extra.change_A(change["new"])
        self.image_box_dominant.change_A(change["new"])
        self.image2_box_high.change_a(change["new"])

    def on_B_selected(self, change):
        self.B = change["new"]
        self.matrix_box_h.change_b(change["new"])
        self.image_box_tropical.change_B(change["new"])
        self.image_box_extra.change_B(change["new"])
        self.image_box_dominant.change_B(change["new"])
        self.image2_box_high.change_b(change["new"])

    def on_Gamma_selected(self, change):
        self.Gamma = change["new"]
        self.matrix_box_h.change_c(change["new"])
        self.image_box_tropical.change_Gamma(change["new"])
        self.image_box_extra.change_Gamma(change["new"])
        self.image_box_dominant.change_Gamma(change["new"])
        self.image2_box_high.change_c(change["new"])
        
    def on_finalfime_selected(self, change):
        change["new"] = int(change["new"])
        self.finaltime = change["new"]
        self.image_box_tropical.change_finaltime(change["new"])
        self.image_box_extra.change_finaltime(change["new"])
        self.image2_box_high.change_finaltime(change["new"])
        self.image2_box_low.change_finaltime(change["new"])
        
    def on_visualization_selected(self, change):
        self.visualization = change["new"]
        
    def on_new_selected(self, change):
        self.image_box_tropical.print_image()
        self.image_box_extra.print_image()
#        self.image2_box_high.print_image()
#        self.image2_box_low.print_image()
#        self.matrix_box_h.print_html()
#        self.matrix_box_l.print_html()