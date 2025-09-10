"""App command for the lightning-pose CLI."""


def register_parser(subparsers):
    """Register the app command parser."""
    app_parser = subparsers.add_parser(
        "run_app",
        description="Start the lightning-pose web application.",
        usage="litpose app [OPTIONS]",
    )
    app_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the app on. Default is 8080.",
    )
    app_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the app on. Default is 127.0.0.1.",
    )


def handle(args):
    """Handle the app command."""
    import importlib.util
    if not importlib.util.find_spec('litpose_app'):
        import sys
        print(
            "❌ App not installed. To install the app:\n\n    pip install lightning-pose-app\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Import lightning_pose modules only when needed
    from litpose_app.main import run_app

    run_app(args.host, args.port)
