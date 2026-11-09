"""Record approval file effects offline: uv run python -m lectures.03.verify_approval."""
import json
from pathlib import Path
import tempfile

from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent


def main():
    records=[]
    original=tools.LEARNING_LOG_DIR
    try:
        for approved in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                tools.LEARNING_LOG_DIR=Path(directory)/'learning_log'
                model=TestModel(call_tools=['save_learning_log'],seed=3)
                first=agent.run_sync('Save a learning record.',model=model)
                assert isinstance(first.output,DeferredToolRequests)
                assert not tools.LEARNING_LOG_DIR.exists()
                call=first.output.approvals[0]
                final=agent.run_sync(model=model,message_history=first.all_messages(),
                    deferred_tool_results=DeferredToolResults(approvals={call.tool_call_id:approved}))
                files=list(tools.LEARNING_LOG_DIR.glob('*.json'))
                assert len(files)==int(approved)
                records.append({'model':'TestModel (offline simulation)','approved':approved,
                    'files_before_approval':0,'files_after_decision':len(files),
                    'requested_arguments':call.args_as_dict(),'answer':final.output,
                    'saved_records':[json.loads(p.read_text()) for p in files],
                    'messages':json.loads(final.all_messages_json())})
    finally:
        tools.LEARNING_LOG_DIR=original
    output=Path(__file__).with_name('results')
    output.mkdir(exist_ok=True)
    (output/'approval_verification.json').write_text(json.dumps(records,indent=2)+'\n')
    print('Verified: deferred=0 files; denied=0 files; approved=1 matching record.')


if __name__=='__main__':
    main()
