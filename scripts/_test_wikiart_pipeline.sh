#!/bin/bash
cd /home/dev_qw/DataCrawlerAgent
timeout 600 python main.py --domain wikiart --config wikiart_config.json 2>&1 | grep -E "调度|Feedback|Reprocess|Verifier|完成|ERROR|exclude|LP-ceiling|shared_style|检测|pipeline_status|Think.*action=" | head -50
