import logging
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    filters,
)

# Import required libraries for the predictive model
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from joblib import load
import os

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define conversation states
FROM_ADDRESS, FROM_REGION, TO_ADDRESS, TO_REGION, TIME_OF_DAY = range(5)

# Load label encoders and models
models_dir = 'models'
region_encoder = load(os.path.join(models_dir, 'region_encoder.joblib'))
time_encoder = load(os.path.join(models_dir, 'time_encoder.joblib'))
models = {app: load(os.path.join(models_dir, f'{app}_model.joblib')) for app in ['TADA', 'GoJek', 'Zig']}

async def start(update: Update, context) -> int:
    """Starts the conversation and asks the user for their current address."""
    context.user_data.clear()
    await update.message.reply_text(
        "I am Ride Aunty. Come, Aunty will help you compare prices, okay?\n"
        "Where are you now, my dear? Please give me your address."
    )
    return FROM_ADDRESS

async def from_address(update: Update, context) -> int:
    """Stores the current address and asks for the current region."""
    context.user_data["from_address"] = update.message.text

    # Send the photo along with the text
    photo_path = 'URA_Region2.png'  # Update the path with the correct photo file
    with open(photo_path, 'rb') as photo:
        await update.message.reply_photo(photo, caption="Ok, thank you. Now, I'll show you a photo. Tell me which part of Singapore you are from: \nCentral, North, West, South, or East?")

    return FROM_REGION


async def from_region(update: Update, context) -> int:
    """Stores the current region and asks for the destination address."""
    context.user_data["from_region"] = update.message.text
    await update.message.reply_text(
        "Ok, now tell me where you are going. Please give me the address."
    )
    return TO_ADDRESS

async def to_address(update: Update, context) -> int:
    """Stores the destination address and asks for the destination region."""
    context.user_data["to_address"] = update.message.text

    # Send the photo along with the text
    photo_path = 'URA_Region2.png'  # Update the path with the correct photo file
    with open(photo_path, 'rb') as photo:
        await update.message.reply_photo(photo, caption="Alright, now I'll show you another photo. Tell me which part of Singapore you are going to:\n Central, North, West, South, or East?")

    return TO_REGION


async def to_region(update: Update, context) -> int:
    """Stores the destination region and asks for the time of day."""
    context.user_data["to_region"] = update.message.text
    await update.message.reply_text(
        "What time of day is it? (e.g., '8:45 AM', '12:30 PM', '6:00 PM')"
    )
    return TIME_OF_DAY

async def time_of_day(update: Update, context) -> int:
    """Stores the time of day and performs the price prediction."""
    from_region = context.user_data.get("from_region")
    to_region = context.user_data.get("to_region")
    time_of_day = update.message.text.strip()

    # Perform price prediction
    predictions = predict_prices(from_region, to_region, time_of_day)
    if isinstance(predictions, str):
        await update.message.reply_text(predictions)
    else:
        message = "Ok, I got some prices for you:"
        for app, low_price, high_price in predictions:
            message += f"\n{app}: ${low_price} - ${high_price}"
        await update.message.reply_text(message)

    return ConversationHandler.END

async def reset(update: Update, context) -> int:
    """Resets the conversation."""
    context.user_data.clear()
    await update.message.reply_text(
        "Conversation has been reset. Please use /start to begin again."
    )
    return ConversationHandler.END

def predict_prices(from_region, to_region, time_of_day):
    """Predict prices using the trained models."""
    valid_regions = ['Central', 'North', 'East', 'West', 'South']
    valid_times = ['8:45 AM', '12:30 PM', '6:00 PM']
    if from_region.capitalize() not in valid_regions or to_region.capitalize() not in valid_regions or time_of_day not in valid_times:
        return "Invalid input. Please check your region or time input and try again."

    from_region_encoded = region_encoder.transform([from_region.capitalize()])[0]
    to_region_encoded = region_encoder.transform([to_region.capitalize()])[0]
    time_of_day_encoded = time_encoder.transform([time_of_day])[0]

    X_new = pd.DataFrame([[from_region_encoded, to_region_encoded, time_of_day_encoded]],
                         columns=['From Region', 'Destination Region', 'Time of Day'])

    results = []
    for app, model in models.items():
        predicted_cost = model.predict(X_new)[0]
        results.append((app, round(predicted_cost - 1.0, 2), round(predicted_cost + 1.0, 2)))  # Adding some range

    return results

def main() -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token("7072506833:AAFVppTvpWWwllHKHJ16NxQyUoot73rhiBk").build()

    # Add conversation handler with the states FROM_ADDRESS, FROM_REGION, TO_ADDRESS, TO_REGION, and TIME_OF_DAY
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            FROM_ADDRESS: [MessageHandler(filters.TEXT & ~filters.COMMAND, from_address)],
            FROM_REGION: [MessageHandler(filters.TEXT & ~filters.COMMAND, from_region)],
            TO_ADDRESS: [MessageHandler(filters.TEXT & ~filters.COMMAND, to_address)],
            TO_REGION: [MessageHandler(filters.TEXT & ~filters.COMMAND, to_region)],
            TIME_OF_DAY: [MessageHandler(filters.TEXT & ~filters.COMMAND, time_of_day)],
        },
        fallbacks=[CommandHandler("reset", reset)],
    )

    application.add_handler(conv_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
