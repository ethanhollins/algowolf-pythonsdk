import click
import json

from .app import App

@click.group()
def app():
	'''Command line tool for Algo Wolf strategy execution'''

@click.command('run', short_help='Run strategy script.')
@click.argument('package')
@click.option(
	'-sid', '-strategy-id', 'strategy_id', required=True
)
@click.option(
	'-acc', '-account-code', 'account_code', required=True
)
@click.option(
	'-key', '--auth-key', 'key', required=True
)
@click.option(
	'-vars', '--input-variables', 'input_variables'
)
@click.option(
	'-c', '--config', 'config', required=True
)
def run(package, strategy_id, account_code, key, input_variables, config):
	input_variables = json.loads(input_variables)
	config = json.loads(config)

	app = App(config, package, strategy_id, account_code, key)
	app.run(input_variables)


@click.command('backtest', short_help='Backtest strategy script.')
@click.argument('package')
@click.option(
	'-sid', '-strategy-id', 'strategy_id', required=True
)
@click.option(
	'-acc', '-account-code', 'account_code', required=True
)
@click.option(
	'-key', '--auth-key', 'key', required=True
)
@click.option(
	'-f', '--from', '_from', required=True
)
@click.option(
	'-t', '--to', 'to', required=True
)
@click.option(
	'-vars', '--input-variables', 'input_variables'
)
@click.option(
	'-c', '--config', 'config', required=True
)
def backtest(package, strategy_id, account_code, key, _from, to, input_variables, config):
	input_variables = json.loads(input_variables)
	config = json.loads(config)

	app = App(config, package, strategy_id, account_code, key)
	app.backtest(_from, to, 'run', input_variables)


@click.command('compile', short_help='Compile strategy script.')
@click.argument('package')
@click.option(
	'-sid', '-strategy-id', 'strategy_id', required=True
)
@click.option(
	'-acc', '-account-code', 'account_code', required=True
)
@click.option(
	'-key', '--auth-key', 'key', required=True
)
@click.option(
	'-c', '--config', 'config', required=True
)
def compile(package, strategy_id, account_code, key, config):
	config = json.loads(config)

	app = App(config, package, strategy_id, account_code, key)
	app.compile()


app.add_command(run)
app.add_command(backtest)
app.add_command(compile)
app()
