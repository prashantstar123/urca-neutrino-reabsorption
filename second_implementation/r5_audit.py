"""Round-5 audit, written independently. Never imports previous referee scripts.

Reuse declared: cold RMF solvers and thermodynamic tables; author's per-energy
kernel functions for a matched-kernel comparison only. A second route uses
an independently reduced, exact-dispersion two-dimensional phase-space integral.
All new files are confined to this script's directory. Run with -B.
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent          # package root: results/ and code/ live here
sys.path.insert(0, str(ROOT / 'code'))
import rmf_dd2_delta as R
import rmf_fsu_delta as F
import urca_rates as UR
import urca_rel as U
import murca_spectral as M
import nc_scatter as NC
# A legacy imported module inserts another project into sys.path. Do not use it.
sys.path[:] = [p for p in sys.path if 'delta_urca_bulkvisc' not in p]
HC, HB = 197.3269804, 6.582119569e-22
GF, GA, CAB = 1.1663788e-11, 1.26, .97370
GT2 = (GF*CAB)**2*(1+3*GA**2)
V = {(r['eos'],r['x'],r['T']):r for r in json.loads((ROOT/'results/viscosity_lc.json').read_text()) if 'error' not in r}
KIN = {(r['x'],r['T']):r for r in json.loads((ROOT/'results/kinetic_response_fine.json').read_text()) if 'error' not in r}

def dump(name, obj):
    (HERE/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def nodes(n, lo, hi):
    g,w=leggauss(n)
    return lo+(g+1)*(hi-lo)/2, w*(hi-lo)/2

def state(eos,x,T,cold=False):
    if eos=='DD2': mod,n0=R,R.N0
    else:
        F.use(eos); mod,n0=F,F.P['N0']
    n=x*n0
    c,ne,nmu,_,_,mu,ms,ier,_=mod.beta_eq_cont(n,R.baryons()[:2])
    assert ier==1
    pn,pp,pe=[HC*(3*np.pi**2*d)**(1/3) for d in (c['n'],c['p'],ne)]
    en,ep=np.hypot(pn,ms['n']),np.hypot(pp,ms['p'])
    v=V[(eos,x,T)]
    dm=0 if cold else KIN[(x,T)]['dmu_np_T']
    return dict(eos=eos,x=x,T=T,n=n,nn=c['n'],np=c['p'],ne=ne,nmu=nmu,
                pn=pn,pp=pp,pe=pe,mn=ms['n'],mp=ms['p'],en=en,ep=ep,
                mu_n=mu['n'],mu_p=mu['p'],DU=mu['n']-mu['p']-en+ep,
                dm=dm,C=v['C'],B=v['Bnp'],ndpdn=v['ndpdn'],tau_AH=v['R5_f1']['tau_AH_s'])

def elastic(s,E,kind):
    # BRT06 Eq.(10) cross section, stable final electron vacancy.
    Q=R.MN-R.MP
    shift=Q+(s['DU'] if kind in ('C','MF') else 0)
    Ee=np.asarray(E)+shift
    if kind in ('B','MF'):
        den=-np.expm1((shift-(s['mu_n']-s['mu_p']))/s['T'])
        eff=(s['nn']-s['np'])/den
    else: eff=s['nn']
    sigma=1.705e-44*(1+3*GA**2)/4*Ee*np.sqrt(np.maximum(Ee**2-R.ME**2,0))/R.ME**2
    sigma*=1+1.1*np.asarray(E)/R.MN
    return sigma*eff*1e41*expit((Ee-s['pe'])/s['T'])

def own_nc(s,E):
    coupl=(.25+3*(GA/2)**2)*s['en']**2+((.5-2*.231)**2+3*(GA/2)**2)*s['ep']**2
    return GF**2*coupl*s['T']**2*E*(np.pi**2+(E/s['T'])**2)*expit(E/s['T'])/(16*np.pi**3)

def own_murca(s,E):
    # Independently transcribed neutron/proton branches, AH2019 Eqs.(29,30),
    # and five-fermion convolution J5. All rates here in natural units.
    pn,pp,pe,mn,mp,T=s['pn'],s['pp'],s['pe'],s['en'],s['ep'],s['T']
    tn=1 if pn>pp+pe else 1-3*(pp+pe-pn)**2/(8*pp*pe)
    if pn>3*pp+pe: tp=0
    elif pn>3*pp-pe: tp=(3*pp+pe-pn)**2/(pn*pe)
    elif pn>pp+pe: tp=4*(3*pp-pn)/pn
    else: tp=2+3*(2*pp-pn)/pe-3*(pp-pe)**2/(pn*pe)
    common=367/(1152*np.pi**3)*(GF*CAB)**2*GA**2/139.57**4*T**6
    lam=common*(mn**3*mp*pn**4*pp/(pn**2+139.57**2)**2*tn + mn*mp**3*pn*(pn-pp)**4/((pn-pp)**2+139.57**2)**2*tp/7)
    j=lambda z:(z**4+10*np.pi**2*z*z+9*np.pi**4)*expit(-z)/24
    integral=quad(lambda z:z*j(z),0,100,epsabs=1e-10,epsrel=1e-11)[0]
    A=lam/(4*T**6*integral)
    def W(z):
        pol=z**4+10*np.pi**2*T*T*z*z+9*np.pi**4*T**4
        return 2*np.pi**2*A*pol*expit(z/T)/24
    return W(E+s['dm']),W(E-s['dm'])

def exact_cc(s,E,nquad=220):
    """Independent exact-dispersion phase space with constant weak vertex.
    q integral is the intersection length of the lepton and baryon triangles.
    W=GT2/(4 pi^3 E) int dEe Ee int dEn En Ep Dq * Fermi factors.
    nu: xp=xn+E-(Ee-mue)+dmu; inverse decay: xp=xn-E-(Ee-mue)+dmu.
    This does not assume Landau-mass parabolic dispersions.
    """
    T,dm,pe=s['T'],s['dm'],s['pe']
    xe,we=nodes(nquad,max(-pe,-40*T-E),40*T+E)
    xn,wn=nodes(nquad,max(s['mn']-s['en'],-40*T-E),40*T+E)
    ee=pe+xe[:,None]; en=s['en']+xn[None,:]
    pn=np.sqrt(np.maximum(en*en-s['mn']**2,0))
    result=[]
    for sn in (1,-1):
        xp=xn[None,:]+sn*E-xe[:,None]+dm
        ep=s['ep']+xp
        pp=np.sqrt(np.maximum(ep*ep-s['mp']**2,0))
        dq=np.maximum(0,np.minimum(ee+E,pn+pp)-np.maximum(np.abs(ee-E),np.abs(pn-pp)))
        dq=np.where(ep>=s['mp'],dq,0)
        if sn==1: f=expit(-xn[None,:]/T)*expit(xp/T)*expit(xe[:,None]/T)
        else: f=expit(xn[None,:]/T)*expit(-xp/T)*expit(-xe[:,None]/T)
        value=GT2/(4*np.pi**3*E)*np.sum(we[:,None]*wn[None,:]*ee*en*ep*dq*f)
        result.append(float(value))
    return result

def angular_response(s,E,wE,Wn,Wb,Wnc,L=10,f=1,nangle=512):
    # Work in seconds. Solve the angularly discretized collision equation,
    # then apply matter feedback. No call to authors' response/admittance.
    omega=2*np.pi*f*1e3; ck=0 if L is None else 2*np.pi*299792458/(L*1000)
    mus,wm=leggauss(nangle); wm/=2
    Lam=0j; chi=0
    for sign,W in ((1,Wn),(-1,Wb)):
        occ=expit(-(E+sign*s['dm'])/s['T'])
        weights=E**2*occ*(1-occ)/(2*np.pi**2*s['T']*HC**3)
        gam=W/(1-occ)/HB; sig=Wnc/HB
        A=np.sum(wm[None,:]/(gam[:,None]+sig[:,None]-1j*omega+1j*ck*mus[None,:]),axis=1)
        h=gam*A/(1-sig*A)
        Lam+=np.sum(wE*weights*gam*(1-h)); chi+=np.sum(wE*weights)
    z=s['C']**2*Lam.real/abs(s['B']*Lam-1j*omega)**2
    return dict(tau=s['ndpdn']/(omega**2*z),L=L,f=f,Lambda_re=float(Lam.real),Lambda_im=float(Lam.imag),
                chi=float(chi),dYe_per_dmu=abs(Lam)/(omega*s['n']),
                dYe_per_drive=abs(Lam/(-1j*omega+s['B']*Lam))/s['n'],
                LC_per_dmu=chi/s['n'],transparent_static=1/(s['B']*s['n']))

def elastic_table():
    old=json.loads((ROOT/'results/elastic_opacity_check.json').read_text())
    out=[]
    for x in (1.,2.):
        s=state('DD2',x,3.)
        # Independent rho-field identity: rearrangement cancels in Un-Up.
        rho_field=R.solve_fields([s['nn'],s['np']],R.baryons()[:2])
        rho_DU=-2*rho_field[-1][2]*rho_field[2]
        for E in (25.,50.,100.,150.):
            row={k:s[k] for k in ('x','T','DU','pe','en','ep','mn','mp')}
            row.update(E=E,DU_from_rho=rho_DU,turn_on_noMF=s['pe']-(R.MN-R.MP),turn_on_MF=s['pe']-(R.MN-R.MP)-s['DU'])
            row.update({f'lambda_{kind}':float(1/elastic(s,E,kind)) for kind in ('A','B','C','MF')})
            row['full_exact_m']=HC*1e-15/exact_cc(s,E,320)[0]
            row['full_project_m']=HC*1e-15/U.spectra(3,s['mn'],s['mp'],s['pn'],s['pp'],s['pe'],s['dm'],[E/3])['W_abs_nu'][0]
            match=[o for o in old if o['x']==x and o['T']==3 and o['E']==E]
            if match:
                row['saved']={k:(v if not isinstance(v,float) or np.isfinite(v) else str(v)) for k,v in match[0].items()}
            out.append(row)
            print('ELASTIC',x,E,{k:row[k] for k in ('lambda_A','lambda_C','lambda_MF','full_exact_m')},flush=True)
    dump('elastic.json',out)

def kinetic():
    out=[]
    for eos,x,T,cold in [('DD2',1,3,False),('DD2',1,4,False),('DD2',2,3,False),('DD2',2,5,False),('DD2',.5,3,False),('DD2',3,4,False),('IUFSU',1,3,True),('BigApple',1.5,3,True)]:
        s=state(eos,x,T,cold)
        E,we=nodes(160,0,50*T)
        raw=U.spectra(T,s['mn'],s['mp'],s['pn'],s['pp'],s['pe'],s['dm'],E/T)
        mn,mb=own_murca(s,E); sc=own_nc(s,E)
        a,b=raw['W_abs_nu']+mn,raw['W_abs_nubar']+mb
        scans=[angular_response(s,E,we,a,b,sc,L,f) for L,f in [(None,1),(.3,1),(1,1),(5,1),(10,1),(20,1),(10,3)]]
        refined=angular_response(s,E,we,a,b,sc,1,1,1024)
        chi=(T*T/6+s['dm']**2/(2*np.pi**2))/HC**3
        r=dict(state=s,route='independent angular quadrature; project dU input, own mU and NC',responses=scans,
               angular_refined=refined,chi_closed_form=chi,LC_per_MeV=chi/s['n'],ratio_LC=1/(s['B']*chi),
               floor=2*s['ndpdn']*s['B']/(s['C']**2*2*np.pi*1000))
        if eos!='DD2':
            for order in (180,300):
                # Independent exact dispersions at all energy nodes.
                cc=np.array([exact_cc(s,e,order) for e in E])
                rr=angular_response(s,E,we,cc[:,0]+mn,cc[:,1]+mb,sc,10,1)
                r[f'exact_order_{order}']=rr
                print('EOS EXACT',eos,order,rr['tau'],flush=True)
        np.savez_compressed(HERE/f'rates_{eos}_{x:g}_{T:g}.npz',E=E,wE=we,Wn=a,Wb=b,Wnc=sc)
        out.append(r);dump('kinetic.json',out)
        print('KIN',eos,x,T,'tau10',scans[4]['tau'],'LC',r['LC_per_MeV'],'finitek',scans[4]['dYe_per_dmu'],flush=True)

def figure():
    s=state('DD2',2,3)
    E=np.unique(np.r_[np.logspace(0,np.log10(200),160),3.,25.,50.,100.,150.])
    raw=U.spectra(3,s['mn'],s['mp'],s['pn'],s['pp'],s['pe'],s['dm'],E/3)
    mn,mb=own_murca(s,E); sc=own_nc(s,E)
    exact=np.array([exact_cc(s,e,320) for e in (3,15.45,25,50,100,150)])
    dens=E**2/(2*np.pi**2); em=raw['W_abs_nu']*np.exp(-(E+s['dm'])/3)*dens
    mu_spec=mb*np.exp(-(E-s['dm'])/3)*dens
    uniform,uw=nodes(240,0,150)
    rraw=U.spectra(3,s['mn'],s['mp'],s['pn'],s['pp'],s['pe'],s['dm'],uniform/3)
    capmean=np.sum(uw*uniform*rraw['dGec'])/np.sum(uw*rraw['dGec'])
    mean_exact_cc=np.array([exact_cc(s,e,220)[0] for e in uniform])
    spec_exact=uniform**2*mean_exact_cc*np.exp(-(uniform+s['dm'])/3)
    exact_mean=np.sum(uw*uniform*spec_exact)/np.sum(uw*spec_exact)
    Emin=s['pp']/s['ep']*(s['pn']-s['pp']-s['pe'])/(1+s['pp']/s['ep'])
    below=np.sum(uw[uniform<Emin]*rraw['dGec'][uniform<Emin])/np.sum(uw*rraw['dGec'])
    points=[]
    for e in (3.,25.,50.,100.,150.):
        i=np.where(E==e)[0][0]
        points.append(dict(E=e,dU_nu=raw['W_abs_nu'][i]/HC*1e15,mU_nu=mn[i]/HC*1e15,
           full_nu=(raw['W_abs_nu'][i]+mn[i])/HC*1e15,dU_nb=raw['W_abs_nubar'][i]/HC*1e15,
           mU_nb=mb[i]/HC*1e15,NC=sc[i]/HC*1e15,elastic_C=float(elastic(s,e,'C')),elastic_MF=float(elastic(s,e,'MF'))))
    grid=np.logspace(0,np.log10(200),160); Et=grid[np.argmin(abs(grid-3))]
    dump('figure.json',dict(state=s,Emin=Emin,dp=s['pn']-s['pp']-s['pe'],points=points,
         project_capmean=capmean,exact_capmean=exact_mean,frac_below_Emin=float(below),
         circle_energy_used=float(Et),circle_NC=float(own_nc(s,Et)/HC*1e15),
         exact_at_E=[3,15.45,25,50,100,150],exact_cc_inverse_m=(exact/HC*1e15).tolist(),
         spec_detailedbalance_relative=float(np.max(abs(em/raw['dGec']*3-1)))))
    np.savez_compressed(HERE/'figure_arrays.npz',E=E,nu=(raw['W_abs_nu']+mn)/HC*1e15,nb=(raw['W_abs_nubar']+mb)/HC*1e15,NC=sc/HC*1e15,elastic=elastic(s,E,'C'),cap=em,nb_spec=mu_spec,thermal=E**2*expit(-E/3))
    print('FIGURE',capmean,exact_mean,Emin,below,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['elastic','kinetic','figure']);a=p.parse_args()
    for m in (R,F,UR,U,M,NC): assert Path(m.__file__).resolve().is_relative_to(ROOT),m.__file__
    globals()[{'elastic':'elastic_table','kinetic':'kinetic','figure':'figure'}[a.part]]()
