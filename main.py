"""
NexTrade - Korean Stock Trading Signal Alert System

Entry point with CLI interface.
"""

import argparse
import asyncio
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from analysis.indicators import calculate_all_indicators
from analysis.relative_strength import calculate_relative_strength
from analysis.sector_rotation import calculate_sector_momentum, get_sector_momentum_detail
from analysis.signals import generate_signals
from config.watchlist import _get_watchlist, get_all_watchlist_tickers, get_ticker_name_map
from data.cache import OHLCVCache
from data.collector import StockCollector
from data.universe import get_stock_ticker_list, validate_tickers
from ml.features import FEATURE_COLUMNS, build_feature_matrix
from ml.predictor import predict_all, predict_single, load_models
from ml.trainer import save_models, train_classifier, train_regressor
from screening.screener import screen_all
from utils.logger import setup_logging

console = Console()


def cmd_collect(args: argparse.Namespace) -> None:
    """Fetch OHLCV data for watchlist stocks."""
    cache = OHLCVCache()
    collector = StockCollector(cache=cache)
    name_map = get_ticker_name_map()

    # Determine tickers to collect
    if args.ticker:
        tickers = [args.ticker]
    elif args.all_universe:
        console.print("[bold]Fetching full stock universe from KRX...[/bold]")
        tickers = get_stock_ticker_list()
    else:
        tickers = get_all_watchlist_tickers()

    if not tickers:
        console.print("[red]No valid tickers to collect.[/red]")
        return

    console.print(
        f"Collecting OHLCV data for [bold]{len(tickers)}[/bold] stocks..."
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Collecting...", total=len(tickers))

        def on_progress(ticker, index, total, rows_added):
            name = name_map.get(ticker, ticker)
            if rows_added > 0:
                desc = f"{name} ({ticker}): +{rows_added} rows"
            else:
                desc = f"{name} ({ticker}): up to date"
            progress.update(task, advance=1, description=desc)

        results = collector.collect_multiple(
            tickers=tickers,
            force_full=args.force,
            progress_callback=on_progress,
        )

    _print_collection_summary(results)


def cmd_status(args: argparse.Namespace) -> None:
    """Show cache statistics."""
    cache = OHLCVCache()
    stats = cache.get_cache_stats()
    name_map = get_ticker_name_map()

    panel_content = (
        f"Tickers cached: [bold]{stats['ticker_count']}[/bold]\n"
        f"Total rows:     [bold]{stats['total_rows']}[/bold]\n"
        f"Date range:     {stats['oldest_date']} ~ {stats['newest_date']}"
    )
    console.print(Panel(panel_content, title="Cache Status", border_style="blue"))

    if args.verbose:
        cached_tickers = cache.get_cached_tickers()
        table = Table(title="Cached Stocks")
        table.add_column("Ticker", style="cyan")
        table.add_column("Name")
        table.add_column("Last Date", style="green")

        for ticker in cached_tickers:
            last_date = cache.get_last_cached_date(ticker)
            name = name_map.get(ticker, "")
            table.add_row(ticker, name, str(last_date) if last_date else "N/A")

        console.print(table)


def cmd_watchlist(args: argparse.Namespace) -> None:
    """Display the stock watchlist."""
    wl = _get_watchlist()
    for key, category in wl.items():
        table = Table(title=f"{category.name_kr} ({category.name_en})")
        table.add_column("Ticker", style="cyan")
        table.add_column("Name")

        for ticker, name in category.stocks:
            table.add_row(ticker, name)

        console.print(table)
        console.print()


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze cached stock data and generate trading signals."""
    cache = OHLCVCache()
    name_map = get_ticker_name_map()

    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = cache.get_cached_tickers()

    if not tickers:
        console.print("[red]No cached data. Run 'collect' first.[/red]")
        return

    results = []
    for ticker in tickers:
        df = cache.get_ohlcv(ticker)
        if len(df) < 120:
            console.print(
                f"[yellow]Skipping {ticker}: only {len(df)} rows "
                f"(need 120+ for MA120)[/yellow]"
            )
            continue

        df = calculate_all_indicators(df)
        signal = generate_signals(df, ticker=ticker)
        if signal:
            results.append(signal)

    if not results:
        console.print("[yellow]No signals generated.[/yellow]")
        return

    # Sort by score descending
    results.sort(key=lambda s: s.total_score, reverse=True)

    # Summary table
    table = Table(title="Stock Trading Signals")
    table.add_column("Ticker", style="cyan")
    table.add_column("Name")
    table.add_column("Close", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Signal")
    table.add_column("Date")

    signal_styles = {
        "strong_buy": "[bold green]",
        "buy": "[green]",
        "neutral": "[white]",
        "sell": "[red]",
        "strong_sell": "[bold red]",
    }

    for sig in results:
        name = name_map.get(sig.ticker, sig.ticker)
        style = signal_styles.get(sig.signal_type, "")
        close_end = "[/]" if style else ""
        table.add_row(
            sig.ticker,
            name,
            f"{sig.close:,.0f}",
            f"{style}{sig.total_score}{close_end}",
            f"{style}{sig.summary}{close_end}",
            sig.date,
        )

    console.print(table)

    # Detail view for specific ticker
    if args.ticker and results:
        sig = results[0]
        console.print()
        _print_signal_detail(sig, name_map)


def cmd_train(args: argparse.Namespace) -> None:
    """Train ML models on cached data."""
    cache = OHLCVCache()

    console.print("[bold]Building feature matrix...[/bold]")
    matrix = build_feature_matrix(cache, add_label=True)

    if matrix.empty:
        console.print("[red]No data available for training. Run 'collect' first.[/red]")
        return

    console.print(
        f"Feature matrix: [bold]{len(matrix)}[/bold] rows, "
        f"[bold]{matrix['ticker'].nunique()}[/bold] tickers"
    )

    # Train classifier
    console.print("\n[bold]Training classifier...[/bold]")
    clf, clf_result = train_classifier(matrix)

    # Train regressor
    console.print("[bold]Training regressor...[/bold]")
    reg, reg_result = train_regressor(matrix)

    # Save models
    features_used = [c for c in FEATURE_COLUMNS if c in matrix.columns]
    meta_path = save_models(clf, reg, clf_result, reg_result, features_used)
    console.print(f"\nModels saved to [cyan]{meta_path.parent}[/cyan]")

    # Display results
    table = Table(title="Training Results")
    table.add_column("Metric", style="bold")
    table.add_column("Classifier", justify="right")
    table.add_column("Regressor", justify="right")

    table.add_row("Samples", str(clf_result.n_samples), str(reg_result.n_samples))
    table.add_row("Features", str(clf_result.n_features), str(reg_result.n_features))
    table.add_row(
        "CV Score (mean±std)",
        f"{clf_result.cv_mean:.3f}±{clf_result.cv_std:.3f}",
        f"{reg_result.cv_mean:.4f}±{reg_result.cv_std:.4f}",
    )
    table.add_row("Accuracy", f"{clf_result.accuracy:.3f}", "-")
    table.add_row("F1 (macro)", f"{clf_result.f1_macro:.3f}", "-")
    table.add_row("MAE", "-", f"{reg_result.mae:.4f}")

    console.print(table)

    # Class distribution
    if clf_result.class_distribution:
        dist = clf_result.class_distribution
        labels = {"0": "Down", "1": "Flat", "2": "Up"}
        dist_str = ", ".join(
            f"{labels.get(k, k)}: {v}" for k, v in sorted(dist.items())
        )
        console.print(f"\nClass distribution: {dist_str}")

    # Top features
    if clf_result.feature_importance:
        sorted_feat = sorted(
            clf_result.feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:10]
        feat_table = Table(title="Top 10 Feature Importance (Classifier)")
        feat_table.add_column("Feature", style="cyan")
        feat_table.add_column("Importance", justify="right")
        for name, imp in sorted_feat:
            feat_table.add_row(name, f"{imp:.0f}")
        console.print(feat_table)


def cmd_predict(args: argparse.Namespace) -> None:
    """Generate ML predictions for stocks."""
    cache = OHLCVCache()
    name_map = get_ticker_name_map()

    if args.ticker:
        # Single ticker prediction with detail view
        try:
            classifier, regressor, features = load_models()
        except FileNotFoundError as e:
            console.print(f"[red]{e}. Run 'train' first.[/red]")
            return

        pred = predict_single(args.ticker, cache, classifier, regressor, features)
        if pred is None:
            console.print(f"[red]Cannot generate prediction for {args.ticker}.[/red]")
            return

        _print_prediction_detail(pred, name_map)
    else:
        # All tickers
        try:
            predictions = predict_all(cache)
        except FileNotFoundError as e:
            console.print(f"[red]{e}. Run 'train' first.[/red]")
            return

        if not predictions:
            console.print("[yellow]No predictions generated.[/yellow]")
            return

        _print_prediction_table(predictions, name_map)


def cmd_sector(args: argparse.Namespace) -> None:
    """Show sector rotation / category momentum analysis."""
    cache = OHLCVCache()

    if args.category:
        # Detail view: per-stock momentum within a category
        try:
            details = get_sector_momentum_detail(cache, args.category)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return

        if not details:
            console.print("[yellow]No data for this category. Run 'collect' first.[/yellow]")
            return

        wl = _get_watchlist()
        cat_info = wl[args.category]
        table = Table(title=f"{cat_info.name_kr} ({cat_info.name_en}) - Stock Momentum")
        table.add_column("Rank", justify="right", style="bold")
        table.add_column("Ticker", style="cyan")
        table.add_column("Name")
        table.add_column("5D", justify="right")
        table.add_column("20D", justify="right")
        table.add_column("60D", justify="right")

        for d in details:
            m5 = f"{d['momentum_5d']:+.2%}"
            m20 = f"{d['momentum_20d']:+.2%}"
            m60 = f"{d['momentum_60d']:+.2%}"
            table.add_row(
                str(d["rank"]),
                d["ticker"],
                d["name"],
                _color_pct(m5, d["momentum_5d"]),
                _color_pct(m20, d["momentum_20d"]),
                _color_pct(m60, d["momentum_60d"]),
            )

        console.print(table)
    else:
        # Summary view: category-level momentum
        results = calculate_sector_momentum(cache)
        if not results:
            console.print("[yellow]No momentum data. Run 'collect' first.[/yellow]")
            return

        table = Table(title="Sector Rotation - Category Momentum")
        table.add_column("Rank", justify="right", style="bold")
        table.add_column("Category")
        table.add_column("5D", justify="right")
        table.add_column("20D", justify="right")
        table.add_column("60D", justify="right")
        table.add_column("Rank Change", justify="right")
        table.add_column("Top Stock")

        for sm in results:
            m5 = f"{sm.momentum_5d:+.2%}"
            m20 = f"{sm.momentum_20d:+.2%}"
            m60 = f"{sm.momentum_60d:+.2%}"

            if sm.rank_change_20d > 0:
                rc = f"[green]+{sm.rank_change_20d}[/green]"
            elif sm.rank_change_20d < 0:
                rc = f"[red]{sm.rank_change_20d}[/red]"
            else:
                rc = "-"

            table.add_row(
                str(sm.rank_20d),
                sm.name_kr,
                _color_pct(m5, sm.momentum_5d),
                _color_pct(m20, sm.momentum_20d),
                _color_pct(m60, sm.momentum_60d),
                rc,
                sm.top_stock,
            )

        console.print(table)


def cmd_ranking(args: argparse.Namespace) -> None:
    """Show comprehensive stock ranking."""
    cache = OHLCVCache()

    console.print("[bold]Computing comprehensive ranking...[/bold]")
    results = screen_all(cache)

    if not results:
        console.print("[yellow]No ranking data. Run 'collect' first.[/yellow]")
        return

    # Filter by category if specified
    if args.category:
        results = [r for r in results if r.category == args.category]
        if not results:
            console.print(f"[yellow]No results for category '{args.category}'.[/yellow]")
            return
        # Re-rank within category
        for i, r in enumerate(results):
            r.composite_rank = i + 1

    # Apply --top limit
    if args.top:
        results = results[: args.top]

    has_ml = any(r.ml_predicted_return is not None for r in results)

    table = Table(title="Stock Comprehensive Ranking")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Ticker", style="cyan")
    table.add_column("Name")
    table.add_column("Close", justify="right")
    table.add_column("Composite", justify="right")
    table.add_column("Tech", justify="right")
    if has_ml:
        table.add_column("ML Pred", justify="right")
        table.add_column("ML Dir")
    table.add_column("Sector Rank", justify="right")
    table.add_column("RS Score", justify="right")

    label_kr = {"up": "상승", "flat": "보합", "down": "하락"}
    label_styles = {"up": "[bold green]", "flat": "[white]", "down": "[bold red]"}

    for r in results:
        row = [
            str(r.composite_rank),
            r.ticker,
            r.name,
            f"{r.close:,.0f}",
            f"[bold]{r.composite_score:.1f}[/bold]",
            f"{r.technical_score:.1f}",
        ]
        if has_ml:
            if r.ml_predicted_return is not None:
                row.append(f"{r.ml_predicted_return:+.2%}")
                ls = label_styles.get(r.ml_label, "")
                le = "[/]" if ls else ""
                row.append(f"{ls}{label_kr.get(r.ml_label, '-')}{le}")
            else:
                row.append("-")
                row.append("-")
        row.append(str(r.sector_momentum_rank))
        row.append(_color_pct(f"{r.relative_strength_score:+.2%}", r.relative_strength_score))

        table.add_row(*row)

    console.print(table)


def cmd_serve(args: argparse.Namespace) -> None:
    """Start Telegram bot + scheduler."""
    from notification.scheduler import create_scheduler
    from notification.telegram_bot import build_telegram_app

    console.print("[bold]Starting server...[/bold]")

    try:
        app = build_telegram_app()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    scheduler = create_scheduler(app)

    async def _run():
        scheduler.start()
        console.print("[green]Scheduler started[/green]")
        console.print("[green]Bot polling started. Press Ctrl+C to stop.[/green]")

        # Print scheduled jobs
        for job in scheduler.get_jobs():
            console.print(f"  Job: {job.name} | Next: {job.next_run_time}")

        await app.initialize()
        await app.start()
        await app.updater.start_polling()

        try:
            # Keep running until interrupted
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            scheduler.shutdown(wait=False)
            await app.updater.stop()
            await app.stop()
            await app.shutdown()
            console.print("\n[yellow]Server stopped.[/yellow]")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")


def cmd_report(args: argparse.Namespace) -> None:
    """Send a report manually via Telegram (for testing)."""
    from notification.telegram_bot import build_telegram_app, send_daily_report, send_sector_report

    try:
        app = build_telegram_app()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    report_type = args.type

    async def _send():
        await app.initialize()
        try:
            if report_type == "daily":
                console.print("Sending daily report...")
                await send_daily_report(app)
            elif report_type == "sector":
                console.print("Sending sector report...")
                await send_sector_report(app)
            else:
                console.print(f"[red]Unknown report type: {report_type}[/red]")
                return
            console.print("[green]Report sent successfully.[/green]")
        finally:
            await app.shutdown()

    asyncio.run(_send())


def _color_pct(formatted: str, value: float) -> str:
    """Wrap a formatted percentage string with green/red based on sign."""
    if value > 0:
        return f"[green]{formatted}[/green]"
    elif value < 0:
        return f"[red]{formatted}[/red]"
    return formatted


def _print_prediction_detail(pred, name_map: dict) -> None:
    """Print detailed prediction for a single stock."""
    name = name_map.get(pred.ticker, pred.ticker)

    confidence_styles = {"high": "bold green", "medium": "yellow", "low": "red"}
    label_styles = {"up": "bold green", "flat": "white", "down": "bold red"}
    label_kr = {"up": "상승", "flat": "보합", "down": "하락"}

    conf_style = confidence_styles.get(pred.confidence, "white")
    label_style = label_styles.get(pred.predicted_label, "white")

    lines = [
        f"[bold]{name}[/bold] ({pred.ticker})",
        "",
        f"Predicted direction: [{label_style}]{label_kr[pred.predicted_label]}[/{label_style}]",
        f"Predicted return (5d): [{label_style}]{pred.predicted_return:+.2%}[/{label_style}]",
        f"Confidence: [{conf_style}]{pred.confidence.upper()}[/{conf_style}] ({pred.confidence_score:.1%})",
        "",
        "Class probabilities:",
        f"  Up:   {pred.probabilities['up']:.1%}",
        f"  Flat: {pred.probabilities['flat']:.1%}",
        f"  Down: {pred.probabilities['down']:.1%}",
    ]
    console.print(Panel("\n".join(lines), title="ML Prediction", border_style="blue"))


def _print_prediction_table(predictions: list, name_map: dict) -> None:
    """Print prediction summary table for multiple stocks."""
    table = Table(title="ML Predictions (5-day forward)")
    table.add_column("Ticker", style="cyan")
    table.add_column("Name")
    table.add_column("Direction")
    table.add_column("Pred Return", justify="right")
    table.add_column("Confidence")
    table.add_column("P(Up)", justify="right")
    table.add_column("P(Flat)", justify="right")
    table.add_column("P(Down)", justify="right")

    label_styles = {"up": "[bold green]", "flat": "[white]", "down": "[bold red]"}
    label_kr = {"up": "상승", "flat": "보합", "down": "하락"}
    conf_styles = {"high": "[bold green]", "medium": "[yellow]", "low": "[red]"}

    for pred in predictions:
        name = name_map.get(pred.ticker, pred.ticker)
        ls = label_styles.get(pred.predicted_label, "")
        cs = conf_styles.get(pred.confidence, "")
        le = "[/]" if ls else ""
        ce = "[/]" if cs else ""

        table.add_row(
            pred.ticker,
            name,
            f"{ls}{label_kr.get(pred.predicted_label, pred.predicted_label)}{le}",
            f"{ls}{pred.predicted_return:+.2%}{le}",
            f"{cs}{pred.confidence.upper()}{ce}",
            f"{pred.probabilities['up']:.1%}",
            f"{pred.probabilities['flat']:.1%}",
            f"{pred.probabilities['down']:.1%}",
        )

    console.print(table)


def _print_signal_detail(sig, name_map: dict) -> None:
    """Print detailed signal breakdown for a single stock."""
    name = name_map.get(sig.ticker, sig.ticker)

    signal_styles = {
        "strong_buy": "bold green",
        "buy": "green",
        "neutral": "white",
        "sell": "red",
        "strong_sell": "bold red",
    }
    style = signal_styles.get(sig.signal_type, "white")

    lines = [
        f"[bold]{name}[/bold] ({sig.ticker})",
        f"Date: {sig.date} | Close: {sig.close:,.0f}원",
        f"Score: [{style}]{sig.total_score}/100[/{style}] — [{style}]{sig.summary}[/{style}]",
        "",
    ]

    for s in sig.signals:
        if s.score > 0:
            icon = "[green]+[/green]"
        elif s.score < 0:
            icon = "[red]-[/red]"
        else:
            icon = " "
        score_str = f"{s.score:+d}" if s.score != 0 else "  0"
        lines.append(f"  {icon} {s.name}: {s.description} ({score_str})")

    console.print(Panel("\n".join(lines), title="Signal Detail", border_style=style))


def _print_collection_summary(results: dict) -> None:
    """Print a summary table of collection results."""
    table = Table(title="Collection Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total tickers", str(results["total_tickers"]))
    table.add_row("Successful", f"[green]{results['success_count']}[/green]")
    table.add_row("Failed", f"[red]{results['fail_count']}[/red]")
    table.add_row("Skipped (up-to-date)", str(results["skipped_count"]))
    table.add_row("Rows added", f"[bold]{results['total_rows']}[/bold]")

    console.print(table)

    if results["failed_tickers"]:
        console.print(
            f"[red]Failed tickers: {', '.join(results['failed_tickers'])}[/red]"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="nextrade",
        description="NexTrade - Korean Stock Trading Signal Alert System",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # collect
    collect_parser = subparsers.add_parser(
        "collect", help="Collect OHLCV data for stocks"
    )
    collect_parser.add_argument(
        "--ticker", "-t", type=str, default=None,
        help="Collect data for a single ticker only",
    )
    collect_parser.add_argument(
        "--all-universe", "-a", action="store_true",
        help="Collect all stocks in the KRX universe (not just watchlist)",
    )
    collect_parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force full re-collection (ignore cache)",
    )

    # status
    status_parser = subparsers.add_parser(
        "status", help="Show cache status and statistics"
    )
    status_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-ticker details",
    )

    # watchlist
    subparsers.add_parser("watchlist", help="Display the stock watchlist")

    # analyze
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze stocks and generate trading signals"
    )
    analyze_parser.add_argument(
        "--ticker", "-t", type=str, default=None,
        help="Analyze a single ticker (shows detail view)",
    )

    # train
    subparsers.add_parser("train", help="Train ML prediction models")

    # predict
    predict_parser = subparsers.add_parser(
        "predict", help="Generate ML predictions for stocks"
    )
    predict_parser.add_argument(
        "--ticker", "-t", type=str, default=None,
        help="Predict for a single ticker (shows detail view)",
    )

    # sector
    sector_parser = subparsers.add_parser(
        "sector", help="Show sector rotation / category momentum"
    )
    sector_parser.add_argument(
        "--category", "-c", type=str, default=None,
        help="Show per-stock detail for a specific category (e.g. electronics, pharma)",
    )

    # ranking
    ranking_parser = subparsers.add_parser(
        "ranking", help="Show comprehensive stock ranking"
    )
    ranking_parser.add_argument(
        "--top", "-n", type=int, default=None,
        help="Show only top N results",
    )
    ranking_parser.add_argument(
        "--category", "-c", type=str, default=None,
        help="Filter by category (e.g. electronics, pharma)",
    )

    # serve
    subparsers.add_parser(
        "serve", help="Start Telegram bot + scheduler server"
    )

    # report
    report_parser = subparsers.add_parser(
        "report", help="Send a report manually via Telegram (for testing)"
    )
    report_parser.add_argument(
        "--type", "-t", type=str, required=True,
        choices=["daily", "sector"],
        help="Report type to send",
    )

    return parser


def main() -> None:
    """Main entry point."""
    setup_logging()

    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "collect": cmd_collect,
        "status": cmd_status,
        "watchlist": cmd_watchlist,
        "analyze": cmd_analyze,
        "train": cmd_train,
        "predict": cmd_predict,
        "sector": cmd_sector,
        "ranking": cmd_ranking,
        "serve": cmd_serve,
        "report": cmd_report,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
