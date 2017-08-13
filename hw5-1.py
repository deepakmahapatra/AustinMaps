import pandas
import scipy
import networkx 
import pyomo.environ as pe
import numpy
import math
import pyomo
import geoplotter
from matplotlib import pyplot as plt

class Maps:
	def __init__(self):
		self.g=geoplotter.GeoPlotter()
		self.addresses=pandas.read_csv("addresses.csv")
		self.df=pandas.read_csv("austin.csv")
		#Cleaning the address data set
		def address(df):
				a=tuple(df.split(","))
				a= a[0]
				a=a.replace(" ", "")
				return a
			
		self.addresses.Address=self.addresses.Address.apply(address)
		#calling the CreateNetwork method to create network as per type

		self.CreateNetwork("pe")
		self.CreateNetwork("nx")

	def CreateNetwork(self,type):
		self.df["start_node"] = self.df.kmlgeometry.str.extract('LINESTRING \(([0-9-.]* [0-9-.]*),')
		self.df["end_node"] = self.df.kmlgeometry.str.extract('([0-9-.]* [0-9-.]*)\)')
		end_list=list(tuple(map(float,(x.split(" ")))) for x in self.df.end_node)
		start_list=list(tuple(map(float,(x.split(" ")))) for x in self.df.start_node)
		self.start_list=start_list
		self.end_list=end_list
		self.df["start_lon"]=pandas.to_numeric(pandas.DataFrame(start_list)[0])
		self.df["start_lat"]=pandas.to_numeric(pandas.DataFrame(start_list)[1])
		self.df["end_lon"]=pandas.to_numeric(pandas.DataFrame(end_list)[0])
		self.df["end_lat"]=pandas.to_numeric(pandas.DataFrame(end_list)[1])
		self.df["start_node"] = start_list
		self.df["end_node"]=end_list
		self.startnodehash=map(hash,start_list)
		self.endnodehash=map(hash,end_list)
		self.df["startnodehash"]=self.startnodehash
		self.df["endnodehash"]=self.endnodehash

		if type=="nx":
			self.G=networkx.DiGraph()
			
			self.G.node_styles={}
			self.G.edge_styles={}
			
			self.G.edge_styles={'default' : dict(color = 'b',linewidth=0.2), 'Path':dict(color = 'black',linewidth=4) }
			self.G.node_styles={'default' : dict(color = 'b', marker = None), 'start' : dict(color ='orange', marker = 'o') , 'dest': dict(color ='g', marker = 'o')}
			graph_df=pandas.DataFrame(self.df.SECONDS)
			graph_df["one_way"]=self.df.ONE_WAY
			graph_df["start_node"]=start_list
			graph_df["end_node"]=end_list
			graph_df["startnodehash"]=self.startnodehash
			graph_df["endnodehash"]=self.endnodehash
			
			
			
			def add_edges(graph_df):
				if graph_df.one_way == "B":
					self.G.add_edge(graph_df.start_node,graph_df.end_node,TIME=graph_df.SECONDS)
					self.G.add_edge(graph_df.end_node,graph_df.start_node,TIME=graph_df.SECONDS)
					#print "******",graph_df.start_node,graph_df.end_node,graph_df.SECONDS
				elif graph_df.one_way == "TF":
					self.G.add_edge(graph_df.start_node,graph_df.end_node,TIME=graph_df.SECONDS)
					#print "******",graph_df.start_node,graph_df.end_node,graph_df.SECONDS
				elif graph_df.one_way == "FT":
					self.G.add_edge(graph_df.end_node,graph_df.start_node,TIME=graph_df.SECONDS)
					#print "******",graph_df.start_node,graph_df.end_node,graph_df.SECONDS
					

			graph_df.apply(add_edges,axis=1)

			for i, data in self.df.iterrows():
				self.G.node[data.start_node]['lat']=data.start_lat
				self.G.node[data.start_node]['lon']=data.start_lon
				self.G.node[data.end_node]['lat']=data.end_lat
				self.G.node[data.end_node]['lon']=data.end_lon

		if type=="pe":


			self.G_Cplex=networkx.DiGraph()
			
			
			self.G_Cplex.node_styles={}
			self.G_Cplex.edge_styles={}
			
			self.G_Cplex.edge_styles={'default' : dict(color = 'b',linewidth=0.2), 'Path':dict(color = 'red',linewidth=4) }
			self.G_Cplex.node_styles={'default' : dict(color = 'b', marker = None), 'start' : dict(color ='orange', marker = 'o') , 'dest': dict(color ='g', marker = 'o')}
			
			

				
	def nearest_node(self,address):
		add_lon=self.addresses[(self.addresses.Address==address)].Lon
		lon = add_lon.get_value(add_lon.index[0],'VALUE')
		add_lat=self.addresses[(self.addresses.Address==address)].Lat
		lat = add_lat.get_value(add_lat.index[0],'VALUE')
		add_node_min=numpy.argmin(numpy.sqrt(((self.df.start_lon)-lon)**2+((self.df.start_lat)-lat)**2),axis=0)
		
		lon=self.df.loc[add_node_min,"start_lon"]
		lat=self.df.loc[add_node_min,"start_lat"]
		a=()
		a+=(lon,lat)
		return a

	def getSPNetwork(self,addressFrom,addressTo):
		start = self.nearest_node(addressFrom)
		dest = self.nearest_node(addressTo)
		sp = networkx.dijkstra_path(self.G, start, dest, weight='TIME')
		return sp
	

	def SPCplex (self,startnode1,destnode1):
		
		startnode_coor = self.nearest_node(startnode1)
		destnode_coor = self.nearest_node(destnode1)
		def add_edges(temp):
				if temp.one_way == "B":
					self.G_Cplex.add_edge(temp.startnodehash,temp.endnodehash,TIME=temp.SECONDS)
					self.G_Cplex.add_edge(temp.endnodehash,temp.startnodehash,TIME=temp.SECONDS)
					#print "******",graph_df.start_node,graph_df.end_node,graph_df.SECONDS
				elif temp.one_way == "TF":
					self.G_Cplex.add_edge(temp.startnodehash,temp.endnodehash,TIME=temp.SECONDS)
					#print "******",graph_df.start_node,graph_df.end_node,graph_df.SECONDS
				elif temp.one_way == "FT":
					self.G_Cplex.add_edge(temp.endnodehash,temp.startnodehash,TIME=temp.SECONDS)

		temp=pandas.DataFrame(self.df.SECONDS)
		temp["one_way"]=self.df.ONE_WAY
		temp["start_node"]=self.start_list
		temp["end_node"]=self.end_list
		temp["startnodehash"]=self.startnodehash
		temp["endnodehash"]=self.endnodehash
		
		
		temp.apply(add_edges,axis=1)
		for i, data in self.df.iterrows():
			self.G_Cplex.node[data.startnodehash]['lat']=data.start_lat
			self.G_Cplex.node[data.startnodehash]['lon']=data.start_lon
			self.G_Cplex.node[data.endnodehash]['lat']=data.end_lat
			self.G_Cplex.node[data.endnodehash]['lon']=data.end_lon
		
		startnode=temp[temp.start_node==startnode_coor].startnodehash.unique()
		destnode=temp[temp.end_node==destnode_coor].endnodehash.unique()
		#self.G_Cplex.node[startnode]['style'] = 'start'
			
		#self.G_Cplex.node[destnode]['style']= 'dest' 
		self.m = pe.ConcreteModel()
		

		# Create sets
		self.m.node_set = pe.Set(initialize=sorted(self.G_Cplex.nodes()))
		self.m.arc_set = pe.Set( initialize=sorted(self.G_Cplex.edges()) , dimen=2)

		# Create variables
		self.m.Y = pe.Var(self.m.arc_set, domain=pe.Binary)
		#self.m.arc_set
		# Create objective
		#print self.G_Cplex.edges(data=True)
		#for e in self.m.arc_set:
			#print self.G_Cplex.edge[e[0]][e[1]]['TIME'] 
		def obj_rule(m):
			return sum(m.Y[e] * self.G_Cplex.edge[e[0]][e[1]]['TIME'] for e in m.arc_set)
		
		self.m.OBJ = pe.Objective(rule=obj_rule, sense=pe.minimize)

		# Flow Balance rule
		def flow_bal_rule(m, n):
			preds = self.G_Cplex.predecessors(n)
			succs = self.G_Cplex.successors(n)
			return sum(m.Y[(p,n)] for p in preds) - sum(m.Y[(n,s)] for s in succs) == 0 -1 * int(n == startnode)+1 * int(n == destnode)
		self.m.FlowBal = pe.Constraint(self.m.node_set, rule=flow_bal_rule)

		# Solving the model
		solver = pyomo.opt.SolverFactory('cplex')
		results = solver.solve(self.m, tee=True, keepfiles=False, options_string="mip_tolerances_integrality=1e-9 mip_tolerances_mipgap=0")

		if (results.solver.status != pyomo.opt.SolverStatus.ok):
			logging.warning('Check solver not ok?')
		if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
			logging.warning('Check solver optimality?') 
		#self.m.Y.pprint()

		#Printing the shortest path in map
		cnode = int(startnode)
		#print "Current node",cnode,destnode
		path = []
		path.append(cnode)
		
	
		while cnode != int(destnode):
			for n in self.G_Cplex.successors(cnode):
				if int(self.m.Y[cnode,n].value) == 1:
					cnode = n
					path.append(cnode)
					break
		return path
	#this function is to find the edges included in the path
	def path(self, start, stop,type):
		
		
		if type=="pe":
			
			path = self.SPCplex(start, stop)

			edgesPath=zip(path[:-1],path[1:])
			for i in edgesPath:
				self.G_Cplex.edge[i[0]][i[1]]['style'] = 'Path'
			self.g.drawNetwork(self.G_Cplex)
			self.g.setZoom(-97.8526, 30.2147, -97.6264, 30.4323)
			plt.show()
			
		if type=="nx":
			self.G.node[self.nearest_node(start)]['style'] = 'start'
			
			self.G.node[self.nearest_node(stop)]['style']= 'dest'
			
			path=self.getSPNetwork(start, stop)
			edgesPath=zip(path[:-1],path[1:])
			for i in edgesPath:
				self.G.edge[i[0]][i[1]]['style'] = 'Path'
			self.g.drawNetwork(self.G)
			self.g.setZoom(-97.8526, 30.2147, -97.6264, 30.4323)
			plt.show()
			

		
	
if __name__=="__main__":
	from matplotlib import pyplot as plt
	add_from="HomeSlicePizza"
	add_to="ClayPit"
	C=Maps()
	#print C.G.edges(data=True)
	#print C.addresses.head(2)
	#print C.addresses[(C.addresses.imbalace==1) or (C.addresses.imbalace==-1)]
	#print C.df.dtypes
	#print C.df.loc[11,"start_node"]
	#print C.getSPNetwork(add_to,add_from)
	#print networkx.number_of_edges (C.G)
	#print C.df.head(2)
	#print C.G.edges(1)
	#print C.G_Cplex.edges(data=True)
	#print C.G_Cplex.edges(data=True)
	#print sum(C.G_Cplex.edge[e[0]][e[1]]['TIME'] for e in C.m.arc_set)
	import geoplotter
	g=geoplotter.GeoPlotter()
	#g.drawNetwork(C.G_Cplex)
	#g.clear()
	C.path("EngineeringTeachingCenter","RudysCountryStoreandBar-B-Q","nx")
	g.clear()
	C.path("EngineeringTeachingCenter","HulaHut","pe")
	g.clear()

