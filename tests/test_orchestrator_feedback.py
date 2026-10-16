"""Orchestration regression tests: no LLM calls or real source writes."""
from types import SimpleNamespace
import pytest
from agents.orchestrator_agent import OrchestratorAgent, PlanStep, DomainConfig, register_domain_config

register_domain_config("test", DomainConfig(
    domain="test", agents_desc={}, default_plan_factory="", critical_agents={"schema_discovery"},
    task_builders={}, namespace_map={}))

class Board:
    def __init__(self): self.data = {}
    def write(self, namespace,key,value): self.data[namespace,key] = value
    def read(self, namespace,key,default=None): return self.data.get((namespace,key),default)
    def snapshot(self): return {}
class Bus:
    def __init__(self): self._queues = {}
    def receive(self, recipient):
        messages = list(self._queues.get(recipient,[])); self._queues[recipient] = []; return messages

def make_runner(filename, scenario):
    scope = {'PlanStep': PlanStep, 'run_pipeline': OrchestratorAgent.run_pipeline,
             'decide_next_agent': OrchestratorAgent.decide_next_agent,
             '_should_continue': OrchestratorAgent._should_continue}
    class Role:
        def __init__(self,runner,name): self.runner,self.name = runner,name
        def execute(self,task,upstream_data):
            r = self.runner
            r.events.append({'role':self.name,'context':task.get('reprocess_context')})
            r.attempts[self.name] = r.attempts.get(self.name,0)+1
            if self.name == 'schema_discovery':
                if scenario == 'schema_invalid_partial':
                    return {'status':'partial','schema_audit':{'valid':False}}
                if scenario == 'critical_error': return {'status':'error'}
                return {'status':'success','published':True,'schema_audit':{'valid':True}}
            if self.name == 'builder':
                if scenario == 'retry_once' and r.attempts[self.name] == 1: return {'status':'error'}
                if scenario == 'builder_error': return {'status':'error'}
                r.graph_version += 1
                if task.get('reprocess_context'): r.context_applied = True
                return {'status':'success','version':r.graph_version}
            if self.name == 'formatter':
                r.packaged_version = r.graph_version
                return {'status':'success','version':r.packaged_version}
            if self.name == 'verifier':
                r.verified_version = r.packaged_version
                if scenario == 'persistent_feedback' or (scenario == 'feedback_once' and not r.context_applied):
                    r.message_bus._queues.setdefault('builder',[]).append(SimpleNamespace(
                        sender='verifier',recipient='builder',msg_type='reprocess_request',priority=2,
                        content={'reason':'coverage issue','exclude_relations':[]}))
                return {'status':'partial' if scenario == 'partial_role' else 'success',
                        'version':r.verified_version}
            return {'status':'error'}
    class Runner:
        run_pipeline = scope['run_pipeline']
        decide_next_agent = scope['decide_next_agent']
        _should_continue = scope['_should_continue']
        def __init__(self):
            self.blackboard,self.message_bus = Board(),Bus()
            self.agents = {n:Role(self,n) for n in ('schema_discovery','builder','formatter','verifier','optional')}
        def analyze_task(self,config):
            self.events,self.attempts = [],{}
            self.graph_version,self.packaged_version,self.verified_version = 0,0,0
            self.context_applied = False
            self.plan = [PlanStep('schema_discovery','declare'),
                         PlanStep('builder','build',['schema_discovery']),
                         PlanStep('formatter','package',['builder']),PlanStep('verifier','check',['formatter'])]
            if scenario == 'optional_error': self.plan.append(PlanStep('optional','optional'))
            return self.plan
        def _build_agent_task(self,name,config,overrides): return dict(overrides)
        def _get_upstream_data(self,completed): return {}
        def _build_final_output(self,result,config):
            result['final_output'] = {'graph_version':self.graph_version,
                                     'packaged_version':self.packaged_version,'verified_version':self.verified_version}
        def should_retry(self,step,result):
            return scenario == 'retry_once' and step.max_retries > 0
    return Runner()

@pytest.mark.parametrize("scenario", (
    "clean", "feedback_once", "persistent_feedback", "retry_once", "schema_invalid_partial",
    "critical_error", "builder_error", "optional_error", "partial_role"))
def test_orchestration_states_and_feedback(scenario):
    runner = make_runner('orchestrator_after.py',scenario)
    result = runner.run_pipeline({'domain':'test'})
    names = [e['role'] for e in runner.events]
    if scenario in ('clean','feedback_once','retry_once'):
        assert result['status'] == 'success', (scenario,result)
        assert runner.graph_version == runner.packaged_version == runner.verified_version
    if scenario == 'feedback_once':
        assert names == ['schema_discovery','builder','formatter','verifier','builder','formatter','verifier']
        assert runner.events[4]['context']['reason'] == 'coverage issue'
        assert len(runner.plan) == 4
    if scenario == 'persistent_feedback':
        assert result['status'] == 'partial'
        assert names.count('builder') == names.count('formatter') == names.count('verifier') == 3
        assert len(result['unresolved_reprocessing']) == 1
    if scenario in ('schema_invalid_partial','critical_error'):
        assert names == ['schema_discovery']
        assert result['status'] == 'failed'
    if scenario == 'builder_error':
        assert names == ['schema_discovery','builder']
        assert result['status'] == 'failed'
        assert 'No dependency-ready stage' in result['errors'][-1]
    if scenario in ('optional_error','partial_role'): assert result['status'] == 'partial'

def test_feedback_budget_resets_between_runs():
    runner = make_runner("unused", "feedback_once")
    for _ in range(2):
        result = runner.run_pipeline({"domain":"test"})
        assert result["status"] == "success"
        assert len(runner.events) == 7
