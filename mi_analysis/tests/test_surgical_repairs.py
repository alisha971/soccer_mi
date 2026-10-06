"""Regression checks for R03/R09/R10/R12/R15 on the new benchmark analysis."""
import numpy as np
from mi_src import graph, scrub, circuits, interventions, context_edits, mi_metrics

def run_all(context):
    rows=[]
    def check(name, test):
        try:
            detail=test(); rows.append({'test':name,'passed':True,'detail':detail})
        except Exception as exc:
            rows.append({'test':name,'passed':False,'detail':repr(exc)})
    def canonical():
        edges=[('factor_player','L0MLP'),('embedding','L0MLP'),('L0ATTN','L1H0.q'),('L0H0','L1H0.q')]
        canonical=graph.canonical_edges(edges,4,2)
        assert len(canonical)==3 and len(set(canonical))==3
        assert graph.kept_atoms(edges,4,2)==graph.kept_atoms(canonical,4,2)
        universe=graph.canonical_edges(graph.all_edges(4,2),4,2)
        for selected in graph.random_edge_sets(universe,len(canonical),100,0):
            assert len(graph.canonical_edges(selected,4,2))==len(canonical)
        return 'Aliases preserve the graph; all 100 random controls match actual connection count'
    check('R09 canonical graph control size',canonical)
    def assignments():
        table=context['table'].iloc[:24].reset_index(drop=True)
        table=table.assign(toy_class=[0]*8+[1]*8+list(range(2,10)))
        spec={'L0MLP':['player','toy_class'],'L1MLP':['player']}
        def remove(inputs,positions):return interventions.remove_feature(inputs,'player',positions)
        kwargs=dict(model=context['model'],bundle=context['bundle'],view=context['view'],table=table,spec=spec,counterfactual=remove,counterfactual_feature='player',draws=3,seed=107,batch_size=8,wrong_class=True)
        _,_,info=scrub.averaged_scrub(**kwargs)
        actual=np.ones(len(table),bool)
        for plan in info['assignments']:
            for component in spec:actual &= plan[component]!=np.arange(len(table))
        assert np.array_equal(actual,info['nonself_every_keyed_node_all_draws'])
        original=[{c:v.copy() for c,v in plan.items()} for plan in info['assignments']]
        plan=[{c:v.copy() for c,v in x.items()} for x in original]
        for d,a in enumerate(plan):a['L0MLP']=circuits.class_donors([0]*len(table),np.random.default_rng(900+d))
        kwargs.update(spec={**spec,'L0MLP':None},wrong_class=False,assignment_plan=plan,modes_override=info['modes'])
        _,_,edited=scrub.averaged_scrub(**kwargs)
        for d,a in enumerate(edited['assignments']):
            assert np.array_equal(a['L1MLP'],original[d]['L1MLP'])
            assert np.array_equal(info['assignments'][d]['L0MLP'],original[d]['L0MLP'])
        assert edited['modes']==info['modes']
        assert edited['draw_clean'].shape==(3,len(table))
        return 'Actual wrong-class donor mask matches every draw; changing one node preserves all other donors and branch modes'
    check('R10/R12 donor masks and coupled node controls',assignments)
    def bounds():
        own=np.array([0.,0.2,0.9,1.],dtype=np.float32)
        new,info=context_edits.bounded_shift(own,0.4,1,upper=1.)
        assert (new>=0).all() and (new<=1).all() and info['clipped'].sum()==2
        new,info=context_edits.bounded_shift(own,0.4,-1)
        assert (new>=0).all() and info['clipped'].sum()==2
        assert not info['changed'][0]
        return 'Support, clipped counts and boundary no-ops agree for both shift directions'
    check('R15 bounded continuous shifts',bounds)
    def uncertainty():
        s=mi_metrics.summarise([1.,2.,3.],[5,5,5],100)
        r=mi_metrics.pooled_recovery([1.,2.],[2.,3.],[5,5],100)
        assert np.isnan(s['ci_low']) and np.isnan(s['ci_high']) and s['mean']==2.
        assert np.isnan(r['ci_low']) and np.isnan(mi_metrics.bootstrap_pvalue([1.,2.],[5,5],100))
        return 'One-match points remain descriptive; no artificial CI or inferential p-value'
    check('R03 one-cluster uncertainty',uncertainty)
    return rows
