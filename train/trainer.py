import torch 

def train(epochs, device,datasets,model,criterion,optimizer,scheduler,metrics,nb_labels):
    print("Training")
    train_loader,valid_loader = datasets
    train_metrics = metrics(nb_labels)
    valid_metrics = metrics(nb_labels)

    complete_log = ""
    for e in range(epochs):  
        model.train()

        train_metrics.init_buffer()

        for inputs, labels, _, info in train_loader:
            inputs = inputs.squeeze().to(device)
            labels = labels.squeeze().float().to(device)


            if inputs.ndim == 3:
                inputs = inputs.unsqueeze(0)

            optimizer.zero_grad()
            outputs = model(inputs)[0]
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_metrics.update(outputs.detach().cpu(),labels.detach().cpu(),loss.detach().cpu(),info)
        
        train_metrics.merge()
        scheduler.step()


        
        model.eval()
        valid_metrics.init_buffer()

        with torch.no_grad():
            for inputs, labels, _, info in valid_loader:
                inputs = inputs.squeeze().to(device)
                labels = labels.squeeze().float().to(device)

                if inputs.ndim == 3:
                    inputs = inputs.unsqueeze(0)

                outputs = model(inputs)[0]
                outputs = outputs.squeeze()

                valid_loss = criterion(outputs, labels)

                valid_metrics.update(outputs.detach().cpu(),labels.detach().cpu(),valid_loss.detach().cpu(),info)

        valid_metrics.merge()

        
        

        log_str = (
            f"{e:02d} | "
            f"Train Loss: {train_metrics.metrics['loss'][-1]:.4f} | "
            f"Train Acc: {train_metrics.metrics['accuracy'][-1]:.2f} | "
            f"Train Macro: {train_metrics.metrics['macro_accuracy'][-1]:.2f} | "
            f"Valid Loss: {valid_metrics.metrics['loss'][-1]:.4f} | "
            f"Valid Acc: {valid_metrics.metrics['accuracy'][-1]:.2f} | "
            f"Valid Macro: {valid_metrics.metrics['macro_accuracy'][-1]:.2f}"
        )
        complete_log += log_str + "\n"
        print(log_str)
        
    return model, (train_metrics, valid_metrics), complete_log