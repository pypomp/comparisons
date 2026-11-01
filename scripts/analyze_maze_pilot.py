"""Plot saved maze cost checks; this is not a fitted localization comparison."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    data=json.loads(Path(a.input).read_text());assert data['passed'] and data['fitted_models']==0
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.7),gridspec_kw={'width_ratios':[1,1.2]})
    names=['Native DPF','Native Stop','Public α=0.97']
    seconds=[r['forward_backward_seconds'] for r in data['cost_records']]
    bars=axes[0].bar(names,seconds,color=['#617c96','#d19b45','#418d85'])
    axes[0].bar_label(bars,fmt='%.2f s',padding=4);axes[0].set_ylim(0,max(seconds)*1.22)
    axes[0].set_ylabel('Seconds per unchanged-state call')
    axes[0].set_title('One full B64 forward/backward call per arm')
    vals=[data['projected_seconds_with_reserve']/3600,data['allocation_cap_seconds']/3600,data['remaining_seconds_at_gate']/3600]
    bars=axes[1].barh(['Conservative work forecast','Original allocation cap','Remaining at cost check'],vals,color=['#b26052','#617c96','#a0a7ac'])
    axes[1].bar_label(bars,labels=[f'{v:.3f} h' for v in vals],padding=4)
    axes[1].set_xlim(0,max(vals)*1.17);axes[1].invert_yaxis();axes[1].set_xlabel('GPU hours (forecast or allowance)')
    axes[1].set_title('Original cost rule stops before training')
    fig.suptitle('PyDPF maze feasibility check: no training or held-out result',fontsize=13)
    fig.text(.5,.025,'T=100 observations, J=100. Single calls include first-execution effects; no optimizer-speed ranking.\nActual allocation cost: 1,475 GPU seconds. Memory passes; time budget fails.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.12,1,.93])
    fig.savefig(out/'cost_check.png');fig.savefig(out/'cost_check.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    summary=dict(fitted_models=0,optimizer_steps=0,held_out_evaluations=0,matched_forward_pairs=len(data['control_audit']['comparisons']),finite_control_calls=data['finite_control_calls'],
                 actual_GPU_seconds=data['actual_GPU_seconds'],projected_seconds_with_reserve=data['projected_seconds_with_reserve'],
                 remaining_seconds_at_gate=data['remaining_seconds_at_gate'],memory_pass=data['memory_pass'],time_pass=data['time_pass'],
                 per_arm_single_call_seconds=dict(zip(names,seconds)),scope='Saved full-scale control and cost results only; no fitted localization comparison',PF_calls=0)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
