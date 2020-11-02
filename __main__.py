from .app import cli

# Run CLI
if __name__ == '__main__':
	cli.app.add_command(cli.run)
	cli.app.add_command(cli.backtest)
	cli.app()
